use crate::{
    state::diff_bind::Resettable,
    store::{
        columns::{self, accounts, cc_members, dreps, pools, pots, proposals, slots, utxo},
        EpochTransitionProgress, HistoricalStores, ReadOnlyStore, Snapshot, Store, StoreError,
        TransactionalContext,
    },
};
use amaru_kernel::{
    network::NetworkName, protocol_parameters::ProtocolParameters, EraHistory, Lovelace, Point,
    PoolId, ProposalId, Slot, StakeCredential, TransactionInput,
};

use slot_arithmetic::Epoch;
use std::{
    borrow::{Borrow, BorrowMut},
    cell::{RefCell, RefMut},
    collections::{BTreeMap, BTreeSet, HashSet},
    ops::{Deref, DerefMut},
};

pub struct RefMutAdapter<'a, T> {
    inner: RefMut<'a, T>,
}

impl<'a, T> RefMutAdapter<'a, T> {
    pub fn new(inner: RefMut<'a, T>) -> Self {
        Self { inner }
    }
}

impl<'a, T> Borrow<T> for RefMutAdapter<'a, T> {
    fn borrow(&self) -> &T {
        self.inner.deref()
    }
}

impl<'a, T> BorrowMut<T> for RefMutAdapter<'a, T> {
    fn borrow_mut(&mut self) -> &mut T {
        self.inner.deref_mut()
    }
}

pub struct OwnedOptionWrapper(Option<pools::Row>);

impl Borrow<Option<pools::Row>> for OwnedOptionWrapper {
    fn borrow(&self) -> &Option<pools::Row> {
        &self.0
    }
}

impl BorrowMut<Option<pools::Row>> for OwnedOptionWrapper {
    fn borrow_mut(&mut self) -> &mut Option<pools::Row> {
        &mut self.0
    }
}

pub struct RefMutAdapterMut<'a, T> {
    inner: &'a mut T,
}

impl<'a, T> RefMutAdapterMut<'a, T> {
    pub fn new(inner: &'a mut T) -> Self {
        Self { inner }
    }
}

impl<'a, T> Borrow<T> for RefMutAdapterMut<'a, T> {
    fn borrow(&self) -> &T {
        self.inner
    }
}

impl<'a, T> BorrowMut<T> for RefMutAdapterMut<'a, T> {
    fn borrow_mut(&mut self) -> &mut T {
        self.inner
    }
}

pub struct MemoryStore {
    tip: RefCell<Option<Point>>,
    epoch_progress: RefCell<Option<EpochTransitionProgress>>,
    utxos: RefCell<BTreeMap<TransactionInput, Option<utxo::Value>>>,
    accounts: RefCell<BTreeMap<StakeCredential, Option<accounts::Row>>>,
    pools: RefCell<BTreeMap<PoolId, Option<pools::Row>>>,
    pots: RefCell<pots::Row>,
    slots: RefCell<BTreeMap<Slot, Option<slots::Row>>>,
    dreps: RefCell<BTreeMap<StakeCredential, Option<dreps::Row>>>,
    proposals: RefCell<BTreeMap<ProposalId, Option<proposals::Row>>>,
    cc_members: RefCell<BTreeMap<StakeCredential, Option<cc_members::Row>>>,
    p_params: RefCell<BTreeMap<Epoch, ProtocolParameters>>,
    era_history: EraHistory,
}

impl MemoryStore {
    pub fn new(era_history: EraHistory) -> Self {
        MemoryStore {
            tip: RefCell::new(None),
            epoch_progress: RefCell::new(None),
            utxos: RefCell::new(BTreeMap::new()),
            accounts: RefCell::new(BTreeMap::new()),
            pools: RefCell::new(BTreeMap::new()),
            pots: RefCell::new(pots::Row::default()),
            slots: RefCell::new(BTreeMap::new()),
            dreps: RefCell::new(BTreeMap::new()),
            proposals: RefCell::new(BTreeMap::new()),
            cc_members: RefCell::new(BTreeMap::new()),
            p_params: RefCell::new(BTreeMap::new()),
            era_history,
        }
    }
}

impl Snapshot for MemoryStore {
    fn epoch(&self) -> Epoch {
        Epoch::from(10)
    }
}

impl ReadOnlyStore for MemoryStore {
    fn get_protocol_parameters_for(&self, epoch: &Epoch) -> Result<ProtocolParameters, StoreError> {
        let map = self.p_params.borrow();
        let params = map.get(epoch).cloned().unwrap_or_default();
        Ok(params)
    }

    fn account(
        &self,
        credential: &amaru_kernel::StakeCredential,
    ) -> Result<Option<crate::store::columns::accounts::Row>, crate::store::StoreError> {
        Ok(self
            .accounts
            .borrow()
            .get(credential)
            .and_then(|opt| opt.clone()))
    }

    fn pool(
        &self,
        pool: &amaru_kernel::PoolId,
    ) -> Result<Option<crate::store::columns::pools::Row>, crate::store::StoreError> {
        Ok(self.pools.borrow().get(pool).and_then(|opt| opt.clone()))
    }

    fn utxo(
        &self,
        input: &amaru_kernel::TransactionInput,
    ) -> Result<Option<amaru_kernel::TransactionOutput>, crate::store::StoreError> {
        Ok(self.utxos.borrow().get(input).and_then(|opt| opt.clone()))
    }

    fn pots(&self) -> Result<crate::summary::Pots, crate::store::StoreError> {
        Ok((&*self.pots.borrow()).into())
    }

    #[allow(refining_impl_trait)]
    fn iter_utxos(
        &self,
    ) -> Result<
        impl Iterator<
            Item = (
                crate::store::columns::utxo::Key,
                crate::store::columns::utxo::Value,
            ),
        >,
        crate::store::StoreError,
    > {
        let utxo_vec: Vec<_> = self
            .utxos
            .borrow()
            .iter()
            .filter_map(|(tx_input, opt_tx_output)| {
                opt_tx_output
                    .as_ref()
                    .map(|tx_output| (tx_input.clone(), tx_output.clone()))
            })
            .collect();
        Ok(utxo_vec.into_iter())
    }

    #[allow(refining_impl_trait)]
    fn iter_block_issuers(
        &self,
    ) -> Result<
        impl Iterator<
            Item = (
                crate::store::columns::slots::Key,
                crate::store::columns::slots::Value,
            ),
        >,
        StoreError,
    > {
        let block_issuer_vec: Vec<_> = self
            .slots
            .borrow()
            .iter()
            .filter_map(|(slot, opt_row)| opt_row.as_ref().map(|row| (slot.clone(), row.clone())))
            .collect();

        Ok(block_issuer_vec.into_iter())
    }

    #[allow(refining_impl_trait)]
    fn iter_pools(
        &self,
    ) -> Result<
        impl Iterator<
            Item = (
                crate::store::columns::pools::Key,
                crate::store::columns::pools::Row,
            ),
        >,
        StoreError,
    > {
        let pool_vec: Vec<_> = self
            .pools
            .borrow()
            .iter()
            .filter_map(|(pool_id, opt_row)| {
                opt_row.as_ref().map(|row| (pool_id.clone(), row.clone()))
            })
            .collect();

        Ok(pool_vec.into_iter())
    }

    #[allow(refining_impl_trait)]
    fn iter_accounts(
        &self,
    ) -> Result<
        impl Iterator<
            Item = (
                crate::store::columns::accounts::Key,
                crate::store::columns::accounts::Row,
            ),
        >,
        StoreError,
    > {
        let accounts_vec: Vec<_> = self
            .accounts
            .borrow()
            .iter()
            .filter_map(|(stake_credential, opt_row)| {
                opt_row
                    .as_ref()
                    .map(|row| (stake_credential.clone(), row.clone()))
            })
            .collect();

        Ok(accounts_vec.into_iter())
    }

    #[allow(refining_impl_trait)]
    fn iter_dreps(
        &self,
    ) -> Result<
        impl Iterator<
            Item = (
                crate::store::columns::dreps::Key,
                crate::store::columns::dreps::Row,
            ),
        >,
        StoreError,
    > {
        let dreps_vec: Vec<_> = self
            .dreps
            .borrow()
            .iter()
            .filter_map(|(stake_credential, opt_row)| {
                opt_row
                    .as_ref()
                    .map(|row| (stake_credential.clone(), row.clone()))
            })
            .collect();

        Ok(dreps_vec.into_iter())
    }

    #[allow(refining_impl_trait)]
    fn iter_proposals(
        &self,
    ) -> Result<
        impl Iterator<
            Item = (
                crate::store::columns::proposals::Key,
                crate::store::columns::proposals::Row,
            ),
        >,
        StoreError,
    > {
        let proposals_vec: Vec<_> = self
            .proposals
            .borrow()
            .iter()
            .filter_map(|(proposal_id, opt_row)| {
                opt_row
                    .as_ref()
                    .map(|row| (proposal_id.clone(), row.clone()))
            })
            .collect();

        Ok(proposals_vec.into_iter())
    }
}

pub struct MemoryTransactionalContext<'a> {
    store: &'a MemoryStore,
}

impl<'a> MemoryTransactionalContext<'a> {
    pub fn new(store: &'a MemoryStore) -> Self {
        Self { store }
    }
}

impl<'a> TransactionalContext<'a> for MemoryTransactionalContext<'a> {
    fn commit(self) -> Result<(), StoreError> {
        Ok(())
    }

    fn rollback(self) -> Result<(), StoreError> {
        Ok(())
    }

    fn try_epoch_transition(
        &self,
        from: Option<EpochTransitionProgress>,
        to: Option<EpochTransitionProgress>,
    ) -> Result<bool, StoreError> {
        let mut progress = self.store.epoch_progress.borrow_mut();

        if *progress != from {
            return Ok(false);
        }
        *progress = to;
        Ok(true)
    }

    fn refund(
        &self,
        credential: &crate::store::columns::accounts::Key,
        deposit: Lovelace,
    ) -> Result<Lovelace, StoreError> {
        let mut accounts = self.store.accounts.borrow_mut();
        match accounts.get_mut(credential) {
            Some(Some(account)) => {
                account.deposit += deposit;
                Ok(0)
            }
            _ => Ok(deposit),
        }
    }

    fn set_protocol_parameters(
        &self,
        epoch: &Epoch,
        protocol_parameters: &ProtocolParameters,
    ) -> Result<(), StoreError> {
        self.store
            .p_params
            .borrow_mut()
            .insert(*epoch, protocol_parameters.clone());
        Ok(())
    }

    fn save(
        &self,
        point: &Point,
        issuer: Option<&crate::store::columns::pools::Key>,
        add: crate::store::Columns<
            impl Iterator<
                Item = (
                    crate::store::columns::utxo::Key,
                    crate::store::columns::utxo::Value,
                ),
            >,
            impl Iterator<Item = crate::store::columns::pools::Value>,
            impl Iterator<
                Item = (
                    crate::store::columns::accounts::Key,
                    crate::store::columns::accounts::Value,
                ),
            >,
            impl Iterator<
                Item = (
                    crate::store::columns::dreps::Key,
                    crate::store::columns::dreps::Value,
                ),
            >,
            impl Iterator<
                Item = (
                    crate::store::columns::cc_members::Key,
                    crate::store::columns::cc_members::Value,
                ),
            >,
            impl Iterator<
                Item = (
                    crate::store::columns::proposals::Key,
                    crate::store::columns::proposals::Value,
                ),
            >,
        >,
        remove: crate::store::Columns<
            impl Iterator<Item = crate::store::columns::utxo::Key>,
            impl Iterator<Item = (crate::store::columns::pools::Key, Epoch)>,
            impl Iterator<Item = crate::store::columns::accounts::Key>,
            impl Iterator<
                Item = (
                    crate::store::columns::dreps::Key,
                    amaru_kernel::CertificatePointer,
                ),
            >,
            impl Iterator<Item = crate::store::columns::cc_members::Key>,
            impl Iterator<Item = crate::store::columns::proposals::Key>,
        >,
        withdrawals: impl Iterator<Item = crate::store::columns::accounts::Key>,
        voting_dreps: BTreeSet<StakeCredential>,
    ) -> Result<(), crate::store::StoreError> {
        let current_tip = self.store.tip.borrow().clone();

        match (point, current_tip) {
            (Point::Specific(new, _), Some(Point::Specific(current, _))) if *new <= current => {
                tracing::trace!(target: "event.store.memory", ?point, "save.point_already_known");
                return Ok(());
            }
            _ => {
                // Update tip
                *self.store.tip.borrow_mut() = Some(point.clone());

                // Add issuer to new slot row
                if let Point::Specific(slot, _) = point {
                    if let Some(issuer) = issuer {
                        self.store.slots.borrow_mut().insert(
                            Slot::from(*slot),
                            Some(crate::store::columns::slots::Row::new(*issuer)),
                        );
                    }
                }
            }
        }

        // Utxos
        for (key, value) in add.utxo {
            self.store.utxos.borrow_mut().insert(key, Some(value));
        }

        // Pools
        for (pool_params, epoch) in add.pools {
            let mut pools = self.store.pools.borrow_mut();
            let key = pool_params.id.clone();

            let updated_row = match pools.get(&key).cloned().flatten() {
                Some(mut row) => {
                    // existing pool → push new future_params
                    row.future_params.push((Some(pool_params.clone()), epoch));
                    row
                }
                None => {
                    // new pool → insert current + push future_param with `None`
                    columns::pools::Row {
                        current_params: pool_params.clone(),
                        future_params: vec![(None, epoch)],
                    }
                }
            };

            pools.insert(key, Some(updated_row));
        }

        // Accounts
        for (key, value) in add.accounts {
            let (delegatee, drep, rewards, deposit) = value;

            let mut row = self
                .store
                .accounts
                .borrow()
                .get(&key)
                .cloned()
                .flatten()
                .unwrap_or_else(|| columns::accounts::Row {
                    delegatee: None,
                    drep: None,
                    rewards: 0,
                    deposit,
                });

            match delegatee {
                Resettable::Set(val) => row.delegatee = Some(val),
                Resettable::Reset => row.delegatee = None,
                Resettable::Unchanged => { /* keep existing */ }
            }

            match drep {
                Resettable::Set(val) => row.drep = Some(val),
                Resettable::Reset => row.drep = None,
                Resettable::Unchanged => { /* keep existing */ }
            }

            if let Some(r) = rewards {
                row.rewards = r;
            }

            // Always overwrite deposit if that's your design
            row.deposit = deposit;

            self.store.accounts.borrow_mut().insert(key, Some(row));
        }

        // Dreps
        let dreps_vec: Vec<_> = add.dreps.collect();

        // Track which DReps are freshly registered
        let newly_registered: HashSet<_> = dreps_vec
            .iter()
            .filter_map(|(cred, (_, register, _))| {
                if register.is_some() {
                    Some(cred.clone())
                } else {
                    None
                }
            })
            .collect();

        // Apply DRep rows to store
        for (credential, (anchor_resettable, register, _epoch)) in &dreps_vec {
            let mut dreps = self.store.dreps.borrow_mut();
            let existing = dreps.get(credential).cloned().flatten();

            let row = if let Some(mut row) = existing {
                if let Some((deposit, registered_at)) = register {
                    row.registered_at = *registered_at;
                    row.deposit = *deposit;
                    row.last_interaction = None;
                }
                anchor_resettable.clone().set_or_reset(&mut row.anchor);
                Some(row)
            } else if let Some((deposit, registered_at)) = register {
                let mut row = columns::dreps::Row {
                    anchor: None,
                    deposit: *deposit,
                    registered_at: *registered_at,
                    last_interaction: None,
                    previous_deregistration: None,
                };
                anchor_resettable.clone().set_or_reset(&mut row.anchor);
                Some(row)
            } else {
                tracing::error!(
                    target: "store::dreps::add",
                    ?credential,
                    "add.register_no_deposit",
                );
                None
            };

            dreps.insert(credential.clone(), row);
        }

        // Update last_interaction for voting DReps, skipping newly registered ones
        for drep in voting_dreps {
            if newly_registered.contains(&drep) {
                continue;
            }

            let slot = point.slot_or_default();
            let current_epoch = self
                .store
                .era_history
                .slot_to_epoch(slot)
                .map_err(|err| StoreError::Internal(err.into()))?;

            if let Some(Some(drep_row)) = self.store.dreps.borrow_mut().get_mut(&drep) {
                drep_row.last_interaction = Some(current_epoch);
            }
        }
        // cc_members
        for (key, value) in add.cc_members {
            let row = match value {
                Resettable::Set(cred) => Some(columns::cc_members::Row {
                    hot_credential: Some(cred),
                }),
                Resettable::Reset => None,
                Resettable::Unchanged => {
                    self.store.cc_members.borrow().get(&key).cloned().flatten()
                }
            };

            self.store.cc_members.borrow_mut().insert(key, row);
        }

        /*
        for (key, value) in add.proposals {
            self.store.proposals.borrow_mut().insert(key, Some(value));
        }*/

        // Delete removed data from each respective column in the store
        for key in remove.utxo {
            self.store.utxos.borrow_mut().remove(&key);
        }

        for (key, _epoch) in remove.pools {
            self.store.pools.borrow_mut().remove(&key);
        }

        for key in remove.accounts {
            self.store.accounts.borrow_mut().remove(&key);
        }

        for (key, _) in remove.dreps {
            self.store.dreps.borrow_mut().remove(&key);
        }

        for key in remove.cc_members {
            self.store.cc_members.borrow_mut().remove(&key);
        }

        /*
        for key in remove.proposals {
            self.store.proposals.borrow_mut().remove(&key);
        }*/

        // Reset rewards data for accounts on withdrawal
        for key in withdrawals {
            if let Some(Some(account)) = self.store.accounts.borrow_mut().get_mut(&key) {
                account.rewards = 0;
            }
        }

        Ok(())
    }

    fn with_pots(
        &self,
        mut with: impl FnMut(Box<dyn BorrowMut<pots::Row> + 'a>),
    ) -> Result<(), StoreError> {
        with(Box::new(RefMutAdapter::new(self.store.pots.borrow_mut())));

        Ok(())
    }

    fn with_pools(&self, mut with: impl FnMut(pools::Iter<'_, '_>)) -> Result<(), StoreError> {
        let mut pools = self.store.pools.borrow_mut();

        let iter = pools.iter_mut().map(|(k, v)| {
            let key = k.clone();
            let boxed: Box<dyn BorrowMut<Option<pools::Row>>> = Box::new(RefMutAdapterMut::new(v));
            (key, boxed)
        });

        with(Box::new(iter));
        Ok(())
    }

    fn with_accounts(
        &self,
        mut with: impl FnMut(accounts::Iter<'_, '_>),
    ) -> Result<(), crate::store::StoreError> {
        let mut pools = self.store.accounts.borrow_mut();

        let iter = pools.iter_mut().map(|(k, v)| {
            let key = k.clone();
            let boxed: Box<dyn BorrowMut<Option<accounts::Row>>> =
                Box::new(RefMutAdapterMut::new(v));
            (key, boxed)
        });
        with(Box::new(iter));
        Ok(())
    }

    fn with_block_issuers(
        &self,
        mut with: impl FnMut(crate::store::columns::slots::Iter<'_, '_>),
    ) -> Result<(), crate::store::StoreError> {
        let mut slots = self.store.slots.borrow_mut();

        let iter = slots.iter_mut().map(|(k, v)| {
            let key = k.clone();
            let boxed: Box<dyn BorrowMut<Option<slots::Row>>> = Box::new(RefMutAdapterMut::new(v));
            (key, boxed)
        });
        with(Box::new(iter));
        Ok(())
    }

    fn with_utxo(
        &self,
        mut with: impl FnMut(crate::store::columns::utxo::Iter<'_, '_>),
    ) -> Result<(), crate::store::StoreError> {
        let mut utxos = self.store.utxos.borrow_mut();

        let iter = utxos.iter_mut().map(|(k, v)| {
            let key = k.clone();
            let boxed: Box<dyn BorrowMut<Option<utxo::Value>>> = Box::new(RefMutAdapterMut::new(v));
            (key, boxed)
        });
        with(Box::new(iter));
        Ok(())
    }

    fn with_dreps(
        &self,
        mut with: impl FnMut(crate::store::columns::dreps::Iter<'_, '_>),
    ) -> Result<(), crate::store::StoreError> {
        let mut dreps = self.store.dreps.borrow_mut();

        let iter = dreps.iter_mut().map(|(k, v)| {
            let key = k.clone();
            let boxed: Box<dyn BorrowMut<Option<dreps::Row>>> = Box::new(RefMutAdapterMut::new(v));
            (key, boxed)
        });
        with(Box::new(iter));
        Ok(())
    }

    fn with_proposals(
        &self,
        mut with: impl FnMut(crate::store::columns::proposals::Iter<'_, '_>),
    ) -> Result<(), crate::store::StoreError> {
        let mut proposals = self.store.proposals.borrow_mut();

        let iter = proposals.iter_mut().map(|(k, v)| {
            let key = k.clone();
            let boxed: Box<dyn BorrowMut<Option<proposals::Row>>> =
                Box::new(RefMutAdapterMut::new(v));
            (key, boxed)
        });
        with(Box::new(iter));
        Ok(())
    }
}

impl Store for MemoryStore {
    fn snapshots(&self) -> Result<Vec<Epoch>, StoreError> {
        Ok(vec![Epoch::from(3)])
    }
    fn next_snapshot(&self, _epoch: Epoch) -> Result<(), crate::store::StoreError> {
        Ok(())
    }
    fn create_transaction(&self) -> impl TransactionalContext<'_> {
        MemoryTransactionalContext { store: self }
    }

    fn tip(&self) -> Result<Point, crate::store::StoreError> {
        Ok(Point::Origin)
    }
}

impl HistoricalStores for MemoryStore {
    fn for_epoch(&self, _epoch: Epoch) -> Result<impl Snapshot, crate::store::StoreError> {
        let era_history: &EraHistory = NetworkName::Preprod.into();
        Ok(MemoryStore::new(era_history.clone()))
    }
}

#[cfg(test)]
impl MemoryStore {
    pub fn get_drep_for_test(&self, credential: &StakeCredential) -> Option<dreps::Row> {
        self.dreps
            .borrow()
            .get(credential)
            .and_then(|opt| opt.clone())
    }

    pub fn get_cc_member_for_test(&self, key: &StakeCredential) -> Option<cc_members::Row> {
        self.cc_members
            .borrow()
            .get(key)
            .and_then(|opt| opt.clone())
    }
}
