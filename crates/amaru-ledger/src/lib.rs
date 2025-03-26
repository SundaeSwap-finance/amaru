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

use amaru_kernel::Point;
use tracing::Span;

pub type RawBlock = Vec<u8>;

#[derive(Clone, Debug)]
pub enum ValidateBlockEvent {
    Validated(Point, RawBlock, Span),
    Rollback(Point),
}

#[derive(Clone, Debug)]
pub enum BlockValidationResult {
    BlockValidated(Point, Span),
    BlockForwardStorageFailed(Point, Span),
    InvalidRollbackPoint(Point),
    RolledBackTo(Point),
}

pub mod context;
pub mod rules;
pub mod state;
pub mod store;
pub mod summary;

#[cfg(test)]
pub(crate) mod test {
    /// Creates a transaction input from a transaction hash str and an index.
    ///
    /// ## Required Imports
    /// ```
    /// use amaru_kernel::{Hash, TransactionInput};
    /// ```
    macro_rules! fake_input {
        ($transaction_id:expr, $index:expr) => {
            TransactionInput {
                transaction_id: Hash::from(hex::decode($transaction_id).unwrap().as_slice()),
                index: $index,
            }
        };
    }

    /// Creates a transaction output with no value, datum, or script ref with a specified address
    ///
    /// ## Required Imports
    /// ```
    ///     use amaru_kernel::{
    ///    Bytes, PostAlonzoTransactionOutput, TransactionOutput, Value,
    /// };
    /// ```
    macro_rules! fake_output {
        ($address:expr) => {
            TransactionOutput::PostAlonzo(PostAlonzoTransactionOutput {
                address: Bytes::from(hex::decode($address).expect("Invalid hex address")),
                value: Value::Coin(0),
                datum_option: None,
                script_ref: None,
            })
        };
    }

    pub(crate) use fake_input;
    pub(crate) use fake_output;
}
