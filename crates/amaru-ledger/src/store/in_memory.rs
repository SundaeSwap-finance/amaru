use crate::store::{
    columns::{accounts, dreps, pools, pots, slots},
    EpochTransitionProgress, HistoricalStores, ReadOnlyStore, Snapshot, Store, StoreError,
    TransactionalContext,
};
use amaru_kernel::{
    protocol_parameters::ProtocolParameters, Lovelace, Point, PoolId, Slot, StakeCredential,
    TransactionInput, TransactionOutput,
};

use slot_arithmetic::Epoch;
use std::collections::BTreeMap;
use std::collections::{BTreeSet, HashMap};

pub struct MemoryStore {
    utxos: BTreeMap<TransactionInput, TransactionOutput>,
    accounts: HashMap<StakeCredential, accounts::Row>,
    pools: HashMap<PoolId, pools::Row>,
    pots: pots::Row,
    slots: BTreeMap<Slot, slots::Row>,
    dreps: HashMap<StakeCredential, dreps::Row>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            utxos: BTreeMap::new(),
            accounts: HashMap::new(),
            pools: HashMap::new(),
            pots: pots::Row::default(),
            slots: BTreeMap::new(),
            dreps: HashMap::new(),
        }
    }
}

impl Snapshot for MemoryStore {
    fn epoch(&self) -> Epoch {
        Epoch::from(10)
    }
}

impl ReadOnlyStore for MemoryStore {
    fn get_protocol_parameters_for(
        &self,
        _epoch: &Epoch,
    ) -> Result<ProtocolParameters, StoreError> {
        Ok(ProtocolParameters::default())
    }

    fn account(
        &self,
        credential: &amaru_kernel::StakeCredential,
    ) -> Result<Option<crate::store::columns::accounts::Row>, crate::store::StoreError> {
        Ok(self.accounts.get(credential).cloned())
    }

    fn pool(
        &self,
        pool: &amaru_kernel::PoolId,
    ) -> Result<Option<crate::store::columns::pools::Row>, crate::store::StoreError> {
        Ok(self.pools.get(pool).cloned())
    }

    fn utxo(
        &self,
        input: &amaru_kernel::TransactionInput,
    ) -> Result<Option<amaru_kernel::TransactionOutput>, crate::store::StoreError> {
        Ok(self.utxos.get(input).cloned())
    }

    fn pots(&self) -> Result<crate::summary::Pots, crate::store::StoreError> {
        Ok((&self.pots).into())
    }

    #[allow(refining_impl_trait)]
    fn iter_utxos(
        &self,
    ) -> Result<
        std::vec::IntoIter<(
            crate::store::columns::utxo::Key,
            crate::store::columns::utxo::Value,
        )>,
        crate::store::StoreError,
    > {
        let utxo_vec: Vec<_> = self
            .utxos
            .iter()
            .map(|(tx_input, tx_output)| (tx_input.clone(), tx_output.clone()))
            .collect::<Vec<_>>();
        Ok(utxo_vec.into_iter())
    }

    #[allow(refining_impl_trait)]
    fn iter_block_issuers(
        &self,
    ) -> Result<
        std::vec::IntoIter<(
            crate::store::columns::slots::Key,
            crate::store::columns::slots::Value,
        )>,
        crate::store::StoreError,
    > {
        let block_issuer_vec: Vec<_> = self
            .slots
            .iter()
            .map(|(slot, row)| (slot.clone(), row.clone()))
            .collect();
        Ok(block_issuer_vec.into_iter())
    }

    #[allow(refining_impl_trait)]
    fn iter_pools(
        &self,
    ) -> Result<
        std::vec::IntoIter<(
            crate::store::columns::pools::Key,
            crate::store::columns::pools::Row,
        )>,
        crate::store::StoreError,
    > {
        let pool_vec: Vec<_> = self
            .pools
            .iter()
            .map(|(pool_id, row)| (pool_id.clone(), row.clone()))
            .collect();
        Ok(pool_vec.into_iter())
    }

    #[allow(refining_impl_trait)]
    fn iter_accounts(
        &self,
    ) -> Result<
        std::vec::IntoIter<(
            crate::store::columns::accounts::Key,
            crate::store::columns::accounts::Row,
        )>,
        crate::store::StoreError,
    > {
        let accounts_vec: Vec<_> = self
            .accounts
            .iter()
            .map(|(stake_credential, row)| (stake_credential.clone(), row.clone()))
            .collect();
        Ok(accounts_vec.into_iter())
    }

    #[allow(refining_impl_trait)]
    fn iter_dreps(
        &self,
    ) -> Result<
        std::vec::IntoIter<(
            crate::store::columns::dreps::Key,
            crate::store::columns::dreps::Row,
        )>,
        crate::store::StoreError,
    > {
        let dreps_vec: Vec<_> = self
            .dreps
            .iter()
            .map(|(stake_credential, row)| (stake_credential.clone(), row.clone()))
            .collect();
        Ok(dreps_vec.into_iter())
    }

    #[allow(refining_impl_trait)]
    fn iter_proposals(
        &self,
    ) -> Result<
        std::vec::IntoIter<(
            crate::store::columns::proposals::Key,
            crate::store::columns::proposals::Row,
        )>,
        crate::store::StoreError,
    > {
        Ok(vec![].into_iter())
    }
}

pub struct MemoryTransactionalContext {}

impl<'a> TransactionalContext<'a> for MemoryTransactionalContext {
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
        _epoch: &Epoch,
        _protocol_parameters: &ProtocolParameters,
    ) -> Result<(), StoreError> {
        Ok(())
    }

    fn save(
        &self,
        _point: &Point,
        _issuer: Option<&crate::store::columns::pools::Key>,
        _add: crate::store::Columns<
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
        _remove: crate::store::Columns<
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
        _withdrawals: impl Iterator<Item = crate::store::columns::accounts::Key>,
        _voting_dreps: BTreeSet<StakeCredential>,
    ) -> Result<(), crate::store::StoreError> {
        Ok(())
    }

    fn with_pots(
        &self,
        _with: impl FnMut(Box<dyn std::borrow::BorrowMut<crate::store::columns::pots::Row> + '_>),
    ) -> Result<(), crate::store::StoreError> {
        Ok(())
    }

    fn with_pools(
        &self,
        _with: impl FnMut(crate::store::columns::pools::Iter<'_, '_>),
    ) -> Result<(), crate::store::StoreError> {
        Ok(())
    }

    fn with_accounts(
        &self,
        _with: impl FnMut(crate::store::columns::accounts::Iter<'_, '_>),
    ) -> Result<(), crate::store::StoreError> {
        Ok(())
    }

    fn with_block_issuers(
        &self,
        _with: impl FnMut(crate::store::columns::slots::Iter<'_, '_>),
    ) -> Result<(), crate::store::StoreError> {
        Ok(())
    }

    fn with_utxo(
        &self,
        _with: impl FnMut(crate::store::columns::utxo::Iter<'_, '_>),
    ) -> Result<(), crate::store::StoreError> {
        Ok(())
    }

    fn with_dreps(
        &self,
        _with: impl FnMut(crate::store::columns::dreps::Iter<'_, '_>),
    ) -> Result<(), crate::store::StoreError> {
        Ok(())
    }

    fn with_proposals(
        &self,
        _with: impl FnMut(crate::store::columns::proposals::Iter<'_, '_>),
    ) -> Result<(), crate::store::StoreError> {
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
        MemoryTransactionalContext {}
    }

    fn tip(&self) -> Result<Point, crate::store::StoreError> {
        Ok(Point::Origin)
    }
}

impl HistoricalStores for MemoryStore {
    fn for_epoch(&self, _epoch: Epoch) -> Result<impl Snapshot, crate::store::StoreError> {
        Ok(MemoryStore {
            utxos: BTreeMap::new(),
            accounts: HashMap::new(),
            pools: HashMap::new(),
            pots: pots::Row::default(),
            slots: BTreeMap::new(),
            dreps: HashMap::new(),
        })
    }
}

#[cfg(test)]
mod in_memory_tests {
    use super::*;
    use amaru_kernel::{
        Bytes, CertificatePointer, Hash, Nullable, PostAlonzoTransactionOutput,
        PseudoTransactionOutput, TransactionInput, TransactionPointer, Value,
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

    fn dummy_pool_row() -> pools::Row {
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
            owners: { Set::from(vec![Hash::from([0u8; 28])]) },
            relays: Vec::new(),
            metadata: Nullable::Null,
        };

        pools::Row {
            current_params: dummy_pool_params,
            future_params: Vec::new(),
        }
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

    #[test]
    fn test_utxo_returns_dummy_output() {
        let mut store = MemoryStore::new();

        let txin = TransactionInput {
            transaction_id: Hash::new([0u8; 32]),
            index: 0,
        };

        let output = PseudoTransactionOutput::PostAlonzo(dummy_post_alonzo_output());

        store.utxos.insert(txin.clone(), output);

        let result = store.utxo(&txin).unwrap();

        assert!(result.is_some(), "UTXO should be found");
    }

    #[test]
    fn test_account_returns_correct_row() {
        let mut store = MemoryStore::new();

        let credential = dummy_stake_credential();

        store
            .accounts
            .insert(credential.clone(), dummy_account_row());

        let result = store.account(&credential).unwrap();

        assert!(result.is_some());

        let row = result.unwrap();
        assert_eq!(row.deposit, 1000);
        assert_eq!(row.rewards, 500);
    }

    #[test]
    fn test_pool_returns_correct_row() {
        let mut store = MemoryStore::new();

        let pool_id = Hash::new([0u8; 28]);
        let dummy_row = dummy_pool_row();

        store.pools.insert(pool_id.clone(), dummy_row);

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
        let mut store = MemoryStore::new();
        store.pots = dummy_pot();

        let result = store.pots().unwrap();

        assert_eq!(result.treasury, 1);
        assert_eq!(result.reserves, 2);
        assert_eq!(result.fees, 3);
    }

    #[test]
    fn test_iter_utxos_returns_inserted_utxos() {
        let mut store = MemoryStore::new();

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

        store.utxos.insert(txin1.clone(), output1.clone());
        store.utxos.insert(txin2.clone(), output2.clone());

        let results = store.iter_utxos().unwrap().collect::<Vec<_>>();

        assert_eq!(results.len(), 2);

        assert_eq!(results[0], (txin1.clone(), output1.clone()));
        assert_eq!(results[1], (txin2.clone(), output2.clone()));
    }

    #[test]
    fn test_iter_accounts_returns_inserted_accounts() {
        let mut store = MemoryStore::new();

        let stake_credential1 = dummy_stake_credential();
        let stake_credential2 = dummy_stake_credential2();
        let row1 = dummy_account_row();
        let row2 = dummy_account_row2();

        store
            .accounts
            .insert(stake_credential1.clone(), row1.clone());
        store
            .accounts
            .insert(stake_credential2.clone(), row2.clone());

        let mut results = store.iter_accounts().unwrap().collect::<Vec<_>>();
        results.sort_by_key(|(k, _)| k.clone());

        let mut expected = vec![(stake_credential1, row1), (stake_credential2, row2)];
        expected.sort_by_key(|(k, _)| k.clone());

        assert_eq!(results, expected);
    }

    #[test]
    fn test_iter_dreps_returns_inserted_dreps() {
        let mut store = MemoryStore::new();

        let stake_credential1 = dummy_stake_credential();
        let stake_credential2 = dummy_stake_credential2();
        let row1 = dummy_drep_row();
        let row2 = dummy_drep_row2();

        store.dreps.insert(stake_credential1.clone(), row1.clone());
        store.dreps.insert(stake_credential2.clone(), row2.clone());

        let mut results = store.iter_dreps().unwrap().collect::<Vec<_>>();
        results.sort_by_key(|(k, _)| k.clone());

        let mut expected = vec![(stake_credential1, row1), (stake_credential2, row2)];
        expected.sort_by_key(|(k, _)| k.clone());

        assert_eq!(results, expected);
    }
}
