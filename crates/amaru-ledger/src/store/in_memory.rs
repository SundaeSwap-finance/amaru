use crate::{
    state::diff_bind::Resettable,
    store::{
        columns::{self, accounts, cc_members, dreps, pools, pots, proposals, slots, utxo},
        EpochTransitionProgress, HistoricalStores, ReadOnlyStore, Snapshot, Store, StoreError,
        TransactionalContext,
    },
};
use amaru_kernel::{
    network::NetworkName, protocol_parameters::ProtocolParameters, CertificatePointer, EraHistory,
    Lovelace, Point, PoolId, ProposalId, Slot, StakeCredential, TransactionInput,
};

use slot_arithmetic::Epoch;
use std::{
    borrow::{Borrow, BorrowMut},
    cell::{RefCell, RefMut},
    collections::{BTreeMap, BTreeSet},
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
        _from: Option<EpochTransitionProgress>,
        _to: Option<EpochTransitionProgress>,
    ) -> Result<bool, StoreError> {
        Ok(true)
    }

    fn refund(
        &self,
        _credential: &crate::store::columns::accounts::Key,
        _deposit: Lovelace,
    ) -> Result<Lovelace, StoreError> {
        Ok(0)
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
        for (credential, (anchor_resettable, register, _epoch)) in add.dreps {
            let mut dreps = self.store.dreps.borrow_mut();
            let existing = dreps.get(&credential).cloned().flatten();
            let mut row = if let Some(mut row) = existing {
                if let Some((deposit, registered_at)) = register {
                    row.registered_at = registered_at;
                    row.deposit = deposit;
                    row.last_interaction = None;
                }
                anchor_resettable.set_or_reset(&mut row.anchor);
                Some(row)
            } else if let Some((deposit, registered_at)) = register {
                let mut row = columns::dreps::Row {
                    anchor: None,
                    deposit,
                    registered_at,
                    last_interaction: None,
                    previous_deregistration: None,
                };
                anchor_resettable.set_or_reset(&mut row.anchor);
                Some(row)
            } else {
                tracing::error!(
                    target: "store::dreps::add",
                    ?credential,
                    "add.register_no_deposit",
                );
                None
            };

            dreps.insert(credential, row);
        }

        // Voting drep interaction updates
        for drep in voting_dreps {
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

        for (key, ptr) in remove.dreps {
            if let Some(Some(row)) = self.store.dreps.borrow_mut().get_mut(&key) {
                row.previous_deregistration = Some(ptr);
            }
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
}
/*
#[cfg(test)]
mod in_memory_tests {
    use super::*;
    use amaru_kernel::{
        protocol_parameters::{
            CostModels, DrepThresholds, PoolThresholds, Prices, ProtocolParametersThresholds,
        },
        Bytes, CertificatePointer, ExUnits, Hash, Nullable, PostAlonzoTransactionOutput,
        PseudoTransactionOutput, RationalNumber, TransactionInput, TransactionPointer, Value,
    };

    fn dummy_post_alonzo_output() -> PostAlonzoTransactionOutput {
        PostAlonzoTransactionOutput {
            address: Bytes::from(vec![0u8; 32]),
            value: Value::Coin(0),
            datum_option: None,
            script_ref: None,
        }
    }

    fn dummy_account_row() -> accounts::Row {
        accounts::Row {
            delegatee: None,
            deposit: 1000,
            drep: None,
            rewards: 500,
        }
    }

    fn dummy_account_row2() -> accounts::Row {
        accounts::Row {
            delegatee: None,
            deposit: 3000,
            drep: None,
            rewards: 200,
        }
    }

    fn dummy_pool_row() -> Option<pools::Row> {
        use amaru_kernel::{Nullable, PoolParams, Set, UnitInterval};
        let dummy_pool_params = PoolParams {
            id: Hash::new([0u8; 28]),
            vrf: Hash::new([0u8; 32]),
            pledge: 0,
            cost: 0,
            margin: UnitInterval {
                numerator: 0,
                denominator: 1,
            },
            reward_account: Bytes::from(vec![0u8; 32]),
            owners: Set::from(vec![Hash::from([0u8; 28])]),
            relays: Vec::new(),
            metadata: Nullable::Null,
        };

        Some(pools::Row {
            current_params: dummy_pool_params,
            future_params: Vec::new(),
        })
    }

    fn dummy_pot() -> pots::Row {
        pots::Row {
            treasury: 1,
            reserves: 2,
            fees: 3,
        }
    }

    fn dummy_stake_credential() -> StakeCredential {
        StakeCredential::AddrKeyhash(Hash::new([0u8; 28]))
    }

    fn dummy_stake_credential2() -> StakeCredential {
        StakeCredential::AddrKeyhash(Hash::new([1u8; 28]))
    }

    fn dummy_drep_row() -> dreps::Row {
        dreps::Row {
            deposit: 500000000,
            anchor: None,
            registered_at: CertificatePointer {
                transaction: TransactionPointer {
                    slot: 100u64.into(),
                    transaction_index: 1,
                },
                certificate_index: 0,
            },
            last_interaction: None,
            previous_deregistration: None,
        }
    }

    fn dummy_drep_row2() -> dreps::Row {
        dreps::Row {
            deposit: 600000000,
            anchor: None,
            registered_at: CertificatePointer {
                transaction: TransactionPointer {
                    slot: 40u64.into(),
                    transaction_index: 5,
                },
                certificate_index: 1,
            },
            last_interaction: None,
            previous_deregistration: None,
        }
    }

    fn dummy_protocol_parameters1() -> ProtocolParameters {
        ProtocolParameters {
            max_block_body_size: 65536,
            max_tx_size: 16384,
            max_header_size: 512,
            max_tx_ex_units: ExUnits {
                mem: 1_000_000,
                steps: 1_000_000_000,
            },
            max_block_ex_units: ExUnits {
                mem: 50_000_000,
                steps: 5_000_000_000,
            },
            max_val_size: 10000,
            max_collateral_inputs: 3,

            // Economic group
            min_fee_a: 44u64,
            min_fee_b: 155381u64,
            stake_credential_deposit: 2000000u64,
            stake_pool_deposit: 500000000u64,
            monetary_expansion_rate: RationalNumber {
                numerator: 1u64,
                denominator: 100u64,
            },
            treasury_expansion_rate: RationalNumber {
                numerator: 1u64,
                denominator: 1000u64,
            },
            coins_per_utxo_byte: 4310u64,
            prices: Prices {
                mem: RationalNumber {
                    numerator: 1u64,
                    denominator: 10u64,
                },
                step: RationalNumber {
                    numerator: 1u64,
                    denominator: 100u64,
                },
            },
            min_fee_ref_script_coins_per_byte: RationalNumber {
                numerator: 1u64,
                denominator: 1000u64,
            },
            max_ref_script_size_per_tx: 10000,
            max_ref_script_size_per_block: 50000,
            ref_script_cost_stride: 500,
            ref_script_cost_multiplier: RationalNumber {
                numerator: 1u64,
                denominator: 1u64,
            },

            // Technical group
            max_epoch: 500u32,
            optimal_stake_pools_count: 500,
            pledge_influence: RationalNumber {
                numerator: 3u64,
                denominator: 10u64,
            },
            collateral_percentage: 150,
            cost_models: CostModels {
                plutus_v1: vec![1; 10],
                plutus_v2: vec![2; 10],
                plutus_v3: vec![3; 10],
            },

            // Governance group
            pool_thresholds: PoolThresholds {
                no_confidence: RationalNumber {
                    numerator: 1,
                    denominator: 5,
                },
                committee: RationalNumber {
                    numerator: 1,
                    denominator: 3,
                },
                committee_under_no_confidence: RationalNumber {
                    numerator: 2,
                    denominator: 5,
                },
                hard_fork: RationalNumber {
                    numerator: 3,
                    denominator: 4,
                },
                security_group: RationalNumber {
                    numerator: 1,
                    denominator: 2,
                },
            },
            drep_thresholds: DrepThresholds {
                no_confidence: RationalNumber {
                    numerator: 1,
                    denominator: 5,
                }, // 20%
                committee: RationalNumber {
                    numerator: 1,
                    denominator: 3,
                }, // 33%
                committee_under_no_confidence: RationalNumber {
                    numerator: 2,
                    denominator: 5,
                }, // 40%
                constitution: RationalNumber {
                    numerator: 3,
                    denominator: 4,
                }, // 75%
                hard_fork: RationalNumber {
                    numerator: 1,
                    denominator: 2,
                }, // 50%
                protocol_parameters: ProtocolParametersThresholds {
                    network_group: RationalNumber {
                        numerator: 1,
                        denominator: 5,
                    }, // 20%
                    economic_group: RationalNumber {
                        numerator: 1,
                        denominator: 4,
                    }, // 25%
                    technical_group: RationalNumber {
                        numerator: 1,
                        denominator: 3,
                    }, // 33%
                    governance_group: RationalNumber {
                        numerator: 1,
                        denominator: 2,
                    }, // 50%
                },
                treasury_withdrawal: RationalNumber {
                    numerator: 1,
                    denominator: 10,
                }, // 10%
            },
            cc_min_size: 3,
            cc_max_term_length: 20u32,
            gov_action_lifetime: 10u32,
            gov_action_deposit: 10000000u64,
            drep_deposit: 2000000u64,
            drep_expiry: 50u32,
        }
    }

    fn dummy_protocol_parameters2() -> ProtocolParameters {
        ProtocolParameters {
            max_block_body_size: 65546,
            max_tx_size: 16394,
            max_header_size: 522,
            max_tx_ex_units: ExUnits {
                mem: 1_000_010,
                steps: 1_000_000_010,
            },
            max_block_ex_units: ExUnits {
                mem: 50_000_010,
                steps: 5_000_000_010,
            },
            max_val_size: 10010,
            max_collateral_inputs: 13,

            min_fee_a: 54u64,
            min_fee_b: 155391u64,
            stake_credential_deposit: 2_000_010u64,
            stake_pool_deposit: 500_000_010u64,
            monetary_expansion_rate: RationalNumber {
                numerator: 11u64,
                denominator: 110u64,
            },
            treasury_expansion_rate: RationalNumber {
                numerator: 11u64,
                denominator: 1010u64,
            },
            coins_per_utxo_byte: 4320u64,
            prices: Prices {
                mem: RationalNumber {
                    numerator: 11u64,
                    denominator: 20u64,
                },
                step: RationalNumber {
                    numerator: 11u64,
                    denominator: 110u64,
                },
            },
            min_fee_ref_script_coins_per_byte: RationalNumber {
                numerator: 11u64,
                denominator: 1010u64,
            },
            max_ref_script_size_per_tx: 10010,
            max_ref_script_size_per_block: 50010,
            ref_script_cost_stride: 510,
            ref_script_cost_multiplier: RationalNumber {
                numerator: 11u64,
                denominator: 11u64,
            },

            max_epoch: 510u32,
            optimal_stake_pools_count: 510,
            pledge_influence: RationalNumber {
                numerator: 13u64,
                denominator: 20u64,
            },
            collateral_percentage: 160,
            cost_models: CostModels {
                plutus_v1: vec![1; 10],
                plutus_v2: vec![2; 10],
                plutus_v3: vec![3; 10],
            },

            pool_thresholds: PoolThresholds {
                no_confidence: RationalNumber {
                    numerator: 11,
                    denominator: 15,
                },
                committee: RationalNumber {
                    numerator: 11,
                    denominator: 13,
                },
                committee_under_no_confidence: RationalNumber {
                    numerator: 12,
                    denominator: 15,
                },
                hard_fork: RationalNumber {
                    numerator: 13,
                    denominator: 14,
                },
                security_group: RationalNumber {
                    numerator: 11,
                    denominator: 12,
                },
            },
            drep_thresholds: DrepThresholds {
                no_confidence: RationalNumber {
                    numerator: 11,
                    denominator: 15,
                },
                committee: RationalNumber {
                    numerator: 11,
                    denominator: 13,
                },
                committee_under_no_confidence: RationalNumber {
                    numerator: 12,
                    denominator: 15,
                },
                constitution: RationalNumber {
                    numerator: 13,
                    denominator: 14,
                },
                hard_fork: RationalNumber {
                    numerator: 11,
                    denominator: 12,
                },
                protocol_parameters: ProtocolParametersThresholds {
                    network_group: RationalNumber {
                        numerator: 11,
                        denominator: 15,
                    },
                    economic_group: RationalNumber {
                        numerator: 11,
                        denominator: 14,
                    },
                    technical_group: RationalNumber {
                        numerator: 11,
                        denominator: 13,
                    },
                    governance_group: RationalNumber {
                        numerator: 11,
                        denominator: 12,
                    },
                },
                treasury_withdrawal: RationalNumber {
                    numerator: 11,
                    denominator: 20,
                },
            },
            cc_min_size: 13,
            cc_max_term_length: 30u32,
            gov_action_lifetime: 20u32,
            gov_action_deposit: 10_000_010u64,
            drep_deposit: 2_000_010u64,
            drep_expiry: 60u32,
        }
    }

    #[test]
    fn test_utxo_returns_dummy_output() {
        let store = MemoryStore::new();

        let txin = TransactionInput {
            transaction_id: Hash::new([0u8; 32]),
            index: 0,
        };

        let output = PseudoTransactionOutput::PostAlonzo(dummy_post_alonzo_output());

        store.utxos.borrow_mut().insert(txin.clone(), Some(output));

        let result = store.utxo(&txin).unwrap();

        assert!(result.is_some(), "UTXO should be found");
    }

    #[test]
    fn test_account_returns_correct_row() {
        let store = MemoryStore::new();

        let credential = dummy_stake_credential();

        // Borrow mutably to insert the dummy account row
        store
            .accounts
            .borrow_mut()
            .insert(credential.clone(), Some(dummy_account_row()));

        // Now call the method that reads immutably
        let result = store.account(&credential).unwrap();

        assert!(result.is_some());

        let row = result.unwrap();
        assert_eq!(row.deposit, 1000);
        assert_eq!(row.rewards, 500);
    }

    #[test]
    fn test_pool_returns_correct_row() {
        let store = MemoryStore::new();

        let pool_id = Hash::new([0u8; 28]);
        let dummy_row = dummy_pool_row();

        // Borrow mutably to insert the dummy row
        store.pools.borrow_mut().insert(pool_id.clone(), dummy_row);

        // Call the method that borrows immutably inside
        let result = store.pool(&pool_id).unwrap();

        assert!(result.is_some());

        let row = result.unwrap();
        assert_eq!(row.current_params.id, pool_id);
        assert_eq!(row.current_params.pledge, 0);
        assert_eq!(row.current_params.cost, 0);
        assert_eq!(row.current_params.margin.numerator, 0);
        assert_eq!(row.current_params.margin.denominator, 1);
        assert_eq!(row.current_params.owners.to_vec().len(), 1);
        assert!(row.current_params.relays.is_empty());
        assert!(matches!(row.current_params.metadata, Nullable::Null));
        assert!(row.future_params.is_empty());
    }

    #[test]
    fn test_pots_returns_correct_data() {
        let store = MemoryStore::new();
        *store.pots.borrow_mut() = dummy_pot();

        let result = store.pots().unwrap();

        assert_eq!(result.treasury, 1);
        assert_eq!(result.reserves, 2);
        assert_eq!(result.fees, 3);
    }

    #[test]
    fn test_iter_utxos_returns_inserted_utxos() {
        let store = MemoryStore::new();

        let txin1 = TransactionInput {
            transaction_id: Hash::new([1u8; 32]),
            index: 0,
        };
        let txin2 = TransactionInput {
            transaction_id: Hash::new([2u8; 32]),
            index: 1,
        };

        let output1 = PseudoTransactionOutput::PostAlonzo(dummy_post_alonzo_output());
        let output2 = PseudoTransactionOutput::PostAlonzo(dummy_post_alonzo_output());

        store
            .utxos
            .borrow_mut()
            .insert(txin1.clone(), Some(output1.clone()));
        store
            .utxos
            .borrow_mut()
            .insert(txin2.clone(), Some(output2.clone()));

        let results = store.iter_utxos().unwrap().collect::<Vec<_>>();

        assert_eq!(results.len(), 2);

        assert_eq!(results[0], (txin1.clone(), output1.clone()));
        assert_eq!(results[1], (txin2.clone(), output2.clone()));
    }

    #[test]
    fn test_iter_accounts_returns_inserted_accounts() {
        let store = MemoryStore::new();

        let stake_credential1 = dummy_stake_credential();
        let stake_credential2 = dummy_stake_credential2();
        let row1 = dummy_account_row();
        let row2 = dummy_account_row2();

        {
            let mut accounts = store.accounts.borrow_mut();
            accounts.insert(stake_credential1.clone(), Some(row1.clone()));
            accounts.insert(stake_credential2.clone(), Some(row2.clone()));
        }

        let mut results = store.iter_accounts().unwrap().collect::<Vec<_>>();
        results.sort_by_key(|(k, _)| k.clone());

        let mut expected = vec![(stake_credential1, row1), (stake_credential2, row2)];
        expected.sort_by_key(|(k, _)| k.clone());

        assert_eq!(results, expected);
    }

    #[test]
    fn test_iter_dreps_returns_inserted_dreps() {
        let store = MemoryStore::new();

        let stake_credential1 = dummy_stake_credential();
        let stake_credential2 = dummy_stake_credential2();
        let row1 = dummy_drep_row();
        let row2 = dummy_drep_row2();

        store
            .dreps
            .borrow_mut()
            .insert(stake_credential1.clone(), Some(row1.clone()));
        store
            .dreps
            .borrow_mut()
            .insert(stake_credential2.clone(), Some(row2.clone()));

        let mut results = store.iter_dreps().unwrap().collect::<Vec<_>>();
        results.sort_by_key(|(k, _)| k.clone());

        let mut expected = vec![(stake_credential1, row1), (stake_credential2, row2)];
        expected.sort_by_key(|(k, _)| k.clone());

        assert_eq!(results, expected);
    }

    fn dummy_proposal_id1() -> ProposalId {
        ProposalId {
            transaction_id: Hash::<32>::from([1u8; 32]),
            action_index: 0,
        }
    }

    fn dummy_proposal_id2() -> ProposalId {
        ProposalId {
            transaction_id: Hash::<32>::from([1u8; 32]),
            action_index: 1,
        }
    }

    fn dummy_proposal_row1() -> proposals::Row {
        proposals::Row {
            proposed_in: ProposalPointer {
                transaction: TransactionPointer {
                    slot: 40u64.into(),
                    transaction_index: 1,
                },
                proposal_index: 21,
            },
            valid_until: 40u64.into(),
            proposal: Proposal {
                deposit: 100,
                reward_account: Bytes::from(vec![0u8; 32]),
                gov_action: GovAction::NoConfidence(Nullable::Some(amaru_kernel::ProposalId {
                    transaction_id: Hash::new([0u8; 32]),
                    action_index: 0,
                })),

                anchor: Anchor {
                    url: "https://example.com/dummy_anchor".to_string(),
                    content_hash: Hash::new([0u8; 32]),
                },
            },
        }
    }

    fn dummy_proposal_row2() -> proposals::Row {
        proposals::Row {
            proposed_in: ProposalPointer {
                transaction: TransactionPointer {
                    slot: 25u64.into(),
                    transaction_index: 2,
                },
                proposal_index: 21,
            },
            valid_until: 47u64.into(),
            proposal: Proposal {
                deposit: 200,
                reward_account: Bytes::from(vec![1u8; 32]),
                gov_action: GovAction::NoConfidence(Nullable::Some(amaru_kernel::ProposalId {
                    transaction_id: Hash::<32>::from([1u8; 32]),
                    action_index: 0,
                })),

                anchor: Anchor {
                    url: "https://example2.com/dummy_anchor".to_string(),
                    content_hash: Hash::<32>::from([1u8; 32]),
                },
            },
        }
    }
    // Test not working due to Ord not being implemented on ProposalId
    #[test]
    fn test_iter_proposals_returns_inserted_proposals() {
        let mut store = MemoryStore::new();

        let proposal_id1 = dummy_proposal_id1();
        let proposal_id2 = dummy_proposal_id2();
        let row1 = dummy_proposal_row1();
        let row2 = dummy_proposal_row2();

        store
            .proposals
            .borrow_mut()
            .insert(proposal_id1.clone(), Some(row1.clone()));
        store
            .proposals
            .borrow_mut()
            .insert(proposal_id2.clone(), Some(row2.clone()));

        let mut results = store.iter_proposals().unwrap().collect::<Vec<_>>();

        let mut expected = vec![(proposal_id1, row1), (proposal_id2, row2)];

        assert_eq!(results, expected);
    }

    #[test]
    fn test_get_protocol_params_returns_params_for_correct_epoch() {
        let store = MemoryStore::new();

        let epoch1: Epoch = 1u64.into();
        let epoch2: Epoch = 2u64.into();

        let protocol_parameters1 = dummy_protocol_parameters1();
        let protocol_parameters2 = dummy_protocol_parameters2();

        store
            .p_params
            .borrow_mut()
            .insert(epoch1.clone(), protocol_parameters1.clone());
        store
            .p_params
            .borrow_mut()
            .insert(epoch2.clone(), protocol_parameters2.clone());

        let result1 = store.get_protocol_parameters_for(&epoch1);
        assert_eq!(result1.unwrap(), protocol_parameters1);

        let result2 = store.get_protocol_parameters_for(&epoch2);
        assert_eq!(result2.unwrap(), protocol_parameters2);

        let epoch3: Epoch = 3u64.into();
        let result3 = store.get_protocol_parameters_for(&epoch3);
        assert!(matches!(result3, ProtocolParamters::default()));
    }

    #[test]
    fn test_set_protocol_params_inserts_params_for_correct_epoch() {
        let store = MemoryStore::new();
        let context = MemoryTransactionalContext { store: &store };

        let epoch: Epoch = 1u64.into();

        let protocol_parameters = dummy_protocol_parameters1();

        context
            .set_protocol_parameters(&epoch, &protocol_parameters)
            .unwrap();

        let result = store.get_protocol_parameters_for(&epoch).unwrap();

        assert_eq!(result, protocol_parameters);
    }
}
*/
