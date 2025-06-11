use crate::store::{
    columns::{accounts, pools, pots, slots, utxo::IterRef},
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
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            utxos: BTreeMap::new(),
            accounts: HashMap::new(),
            pools: HashMap::new(),
            pots: pots::Row::default(),
            slots: BTreeMap::new(),
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
    fn iter_utxos(&self) -> Result<IterRef<'_>, StoreError> {
        Ok(self.utxos.iter())
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
            .map(|(key, row)| (key.clone(), row.clone()))
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
        Ok(vec![].into_iter())
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
        Ok(vec![].into_iter())
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
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use amaru_kernel::Bytes;
    use amaru_kernel::{
        Hash, PostAlonzoTransactionOutput, PseudoTransactionOutput, TransactionInput, Value,
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

        println!("UTXO result: {:?}", result);

        assert!(result.is_some(), "UTXO should be found");
    }

    #[test]
    fn test_account_returns_correct_row() {
        let mut store = MemoryStore::new();

        let key_bytes = [0u8; 28];
        let addr_keyhash = Hash::new(key_bytes);

        let credential = StakeCredential::AddrKeyhash(addr_keyhash);

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
        use amaru_kernel::Nullable;
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
    fn iter_utxos_returns_inserted_utxos() {
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

        // iter_utxos now returns references, so results will be Vec<(&TransactionInput, &PseudoTransactionOutput)>
        let results = store.iter_utxos().unwrap().collect::<Vec<_>>();

        assert_eq!(results.len(), 2);

        // Compare with references using ref pattern or & syntax
        assert_eq!(results[0], (&txin1, &output1));
        assert_eq!(results[1], (&txin2, &output2));
    }

    #[test]
    fn iter_block_issuers_returns_inserted_slots() {
        let mut store = MemoryStore::new();

        let slot1 = Slot::from(1u64);
        let slot2 = Slot::from(2u64);

        let pool_id1 = Hash::new([2u8; 28]);
        let pool_id2 = Hash::new([2u8; 28]);

        let row1 = slots::Row::new(pool_id1);
        let row2 = slots::Row::new(pool_id2);

        store.slots.insert(slot1, row1.clone());
        store.slots.insert(slot2, row2.clone());

        let results = store.iter_block_issuers().unwrap().collect::<Vec<_>>();

        assert_eq!(results.len(), 2);

        assert_eq!(results[0], (Slot::from(1u64), row1));
        assert_eq!(results[1], (Slot::from(2u64), row2));
    }
}
