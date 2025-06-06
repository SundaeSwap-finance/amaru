// Copyright 2024 PRAGMA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use borrowable_proxy::BorrowableProxy;
use minicbor as cbor;
use std::{borrow::BorrowMut, cell::RefCell, marker::PhantomData, rc::Rc};

/// An iterator over borrowable items. This allows to define a Rust-idiomatic API for accessing
/// items in read-only or read-write mode. When provided in a callback, it allows the callee to
/// then perform specific operations (e.g. database updates) on items that have been mutably
/// borrowed.
pub type IterBorrow<'a, 'b, K, V> = Box<dyn Iterator<Item = (K, Box<dyn BorrowMut<V> + 'a>)> + 'b>;

pub fn new<'a, const PREFIX: usize, K: Clone, V: Clone>(
    inner: impl Iterator<Item = (Box<[u8]>, Box<[u8]>)> + 'a,
) -> KeyValueIterator<'a, PREFIX, K, V> {
    KeyValueIterator {
        inner: Box::new(inner),
        updates: Rc::new(RefCell::new(Vec::new())),
        phantom_k: PhantomData,
    }
}

/// A KeyValueIterator defines an abstraction on top of another iterator, that stores updates applied to
/// iterated elements. This allows, for example, to iterate over elements of a key/value store,
/// collect mutations to the underlying value, and persist those mutations without leaking the
/// underlying store implementation to the caller.
pub struct KeyValueIterator<'a, const PREFIX: usize, K, V: Clone> {
    updates: SharedVec<Option<V>>,
    inner: RawIterator<'a>,
    phantom_k: PhantomData<&'a K>,
}

pub type RawIterator<'a> = Box<dyn Iterator<Item = (Box<[u8]>, Box<[u8]>)> + 'a>;

pub type SharedVec<T> = Rc<RefCell<Vec<(Vec<u8>, T)>>>;

impl<const PREFIX: usize, K: Clone, V: Clone> KeyValueIterator<'_, PREFIX, K, V> {
    /// Obtain an iterator on the updates to be done. This takes ownership of the original
    /// iterator to ensure that it is correctly de-allocated as we now process updates.
    pub fn into_iter_updates(self) -> impl Iterator<Item = (Vec<u8>, Option<V>)> {
        // NOTE: In principle, 'into_iter_updates' is only called once all callbacks on the inner
        // iterator have resolved; so the absolute count of strong references should be 1 and no
        // cloning should occur here. We can prove that with the next assertion.
        assert_eq!(Rc::strong_count(&self.updates), 1);
        Rc::unwrap_or_clone(self.updates).into_inner().into_iter()
    }
}

impl<
        'a,
        const PREFIX: usize,
        K: Clone + for<'d> cbor::Decode<'d, ()>,
        V: Clone + for<'d> cbor::Decode<'d, ()>,
    > KeyValueIterator<'a, PREFIX, K, V>
where
    Self: 'a,
{
    pub fn as_iter_borrow(&mut self) -> IterBorrow<'a, '_, K, Option<V>> {
        Box::new(self.filter_map(Result::ok))
    }
}

impl<
        'a,
        const PREFIX: usize,
        K: Clone + for<'d> cbor::Decode<'d, ()>,
        V: Clone + for<'d> cbor::Decode<'d, ()>,
    > Iterator for KeyValueIterator<'a, PREFIX, K, V>
where
    Self: 'a,
{
    type Item = Result<(K, Box<dyn BorrowMut<Option<V>> + 'a>), cbor::decode::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| {
            let key: K = cbor::decode(&k[PREFIX..])?;
            let original: V = cbor::decode(&v)?;

            // NOTE: .clone() is cheap, because `self.updates` is an Rc
            let updates = self.updates.clone();

            let on_update =
                move |new: Option<V>| updates.as_ref().borrow_mut().push((k.to_vec(), new));

            Ok((
                key,
                Box::new(BorrowableProxy::new(Some(original), on_update))
                    as Box<dyn BorrowMut<Option<V>> + 'a>,
            ))
        })
    }
}

pub mod borrowable_proxy {
    use std::borrow::{Borrow, BorrowMut};

    /// A wrapper around an item that allows to perform a specific action when mutated. This is handy
    /// to provide a clean abstraction to layers upstream (e.g. Iterators) while performing
    /// implementation-specific actions (such as updating a persistent storage) when items are mutated.
    ///
    /// More specifically, the provided hook is called whenever the item is mutably borrowed; from
    /// which it may or may not be mutated. But, why would one mutably borrow the item if not to mutate
    /// it?
    pub struct BorrowableProxy<T, F>
    where
        F: FnOnce(T),
    {
        item: Option<T>,
        hook: Option<F>,
        borrowed: bool,
    }

    impl<T, F> BorrowableProxy<T, F>
    where
        F: FnOnce(T),
    {
        pub fn new(item: T, hook: F) -> Self {
            Self {
                item: Some(item),
                hook: Some(hook),
                borrowed: false,
            }
        }
    }

    // Provide a read-only access, through an immutable borrow.
    impl<T, F> Borrow<T> for BorrowableProxy<T, F>
    where
        F: FnOnce(T),
    {
        #[allow(clippy::unwrap_used)]
        fn borrow(&self) -> &T {
            self.item.as_ref().unwrap()
        }
    }

    // Provide a write access, through a mutable borrow.
    impl<T, F> BorrowMut<T> for BorrowableProxy<T, F>
    where
        F: FnOnce(T),
    {
        #[allow(clippy::unwrap_used)]
        fn borrow_mut(&mut self) -> &mut T {
            self.borrowed = true;
            self.item.as_mut().unwrap()
        }
    }

    // Install a handler for the hook when the object is dropped from memory.
    impl<T, F> Drop for BorrowableProxy<T, F>
    where
        F: FnOnce(T),
    {
        fn drop(&mut self) {
            if self.borrowed {
                if let (Some(item), Some(hook)) = (self.item.take(), self.hook.take()) {
                    hook(item);
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn trigger_hook_on_mutation() {
            let mut xs = Vec::new();
            {
                let mut item = BorrowableProxy::new(42, |n| xs.push(n));
                let item_ref: &mut usize = item.borrow_mut();
                *item_ref -= 28;
            }
            assert_eq!(xs, vec![14], "{xs:?}")
        }

        #[test]
        fn ignore_hook_on_simple_borrow() {
            let mut xs: Vec<usize> = Vec::new();
            {
                let item = BorrowableProxy::new(42, |n| xs.push(n));
                let item_ref: &usize = item.borrow();
                assert_eq!(item_ref, &42);
            }
            assert!(xs.is_empty(), "{xs:?}")
        }
    }
}
