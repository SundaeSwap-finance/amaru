use std::collections::BTreeSet;

use crate::state::diff_bind::{self};
use crate::store::columns::accounts::test::any_stake_credential;
use crate::store::columns::pools::tests::any_pool_id;
use crate::store::columns::utxo::test::{any_pseudo_transaction_output, any_txin};
use crate::store::columns::{accounts, dreps, pots, proposals, slots};
use crate::store::in_memory::MemoryStore;
use crate::store::{Columns, StoreError};
use crate::store::{ReadOnlyStore, TransactionalContext};
use amaru_kernel::TransactionOutput;
use amaru_kernel::{
    Anchor, CertificatePointer, DRep, Hash, Point, PoolId, PoolParams, ProposalId, Slot,
    StakeCredential, TransactionInput,
};
use proptest::prelude::Strategy;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;
use slot_arithmetic::Epoch;

fn generate_txin() -> TransactionInput {
    any_txin()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current()
}

fn generate_output() -> TransactionOutput {
    any_pseudo_transaction_output()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current()
}

fn generate_stake_credential() -> StakeCredential {
    any_stake_credential()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current()
}

fn generate_account_row() -> accounts::Row {
    crate::store::accounts::test::any_row()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current()
}

fn generate_pool_id() -> PoolId {
    any_pool_id()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current()
}

fn generate_pool_row() -> (PoolParams, Epoch) {
    let pool_params = crate::store::pools::tests::any_pool_params()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current();

    let epoch = Epoch::from(0u64);

    (pool_params, epoch)
}

fn generate_pots_row() -> pots::Row {
    crate::store::pots::test::any_row()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current()
}

fn generate_slot() -> Slot {
    crate::store::slots::tests::any_slot()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current()
}

fn generate_slot_row() -> slots::Row {
    crate::store::slots::tests::any_row()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current()
}

fn generate_drep_row() -> dreps::Row {
    let mut row = crate::store::dreps::tests::any_row()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current();

    // Ensure anchor is present
    if row.anchor.is_none() {
        row.anchor = Some(dummy_anchor());
    }

    row
}

fn dummy_anchor() -> Anchor {
    Anchor {
        url: "https://example.com".to_string(),
        content_hash: Hash::from([0u8; 32]),
    }
}

fn _generate_proposal_id() -> ProposalId {
    crate::store::proposals::tests::any_proposal_id()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current()
}

fn _generate_proposal_row() -> proposals::Row {
    crate::store::proposals::tests::any_row()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current()
}

fn dummy_delegatee() -> Hash<28> {
    Hash::from([0u8; 28])
}

fn dummy_drep() -> (DRep, CertificatePointer) {
    (
        DRep::Key(Hash::from([0u8; 28])),
        CertificatePointer::default(),
    )
}

pub struct SeededData {
    pub txin: TransactionInput,
    pub output: TransactionOutput,
    pub account_key: StakeCredential,
    pub account_row: accounts::Row,
    pub pool_params: PoolParams,
    pub pool_epoch: Epoch,
    pub drep_key: StakeCredential,
    pub drep_row: dreps::Row,
    //pub proposal_id: ProposalId,
    //pub proposal_row: proposals::Row,
}

pub fn seed_store<'a, C: TransactionalContext<'a>>(
    context: &'a C,
) -> Result<SeededData, StoreError> {
    use diff_bind::Resettable;

    let point = Point::Origin;

    // UTXO
    let txin = generate_txin();
    let output = generate_output();
    let utxos_iter = std::iter::once((txin.clone(), output.clone()));

    // Account
    let account_key = generate_stake_credential();
    let account_key_clone = account_key.clone(); // clone BEFORE it gets moved

    let account_row = generate_account_row();

    // Clone fields explicitly to avoid moving out of account_row
    let delegatee = match &account_row.delegatee {
        Some(pool_id) => Resettable::Set(pool_id.clone()),
        None => Resettable::Reset,
    };

    let drep = match &account_row.drep {
        Some(drep_pair) => Resettable::Set(drep_pair.clone()),
        None => Resettable::Reset,
    };

    let rewards = Some(account_row.rewards); // Copy
    let deposit = account_row.deposit; // Copy

    let accounts_iter = std::iter::once((
        account_key_clone, // consume the clone here
        (delegatee, drep, rewards, deposit),
    ));

    // Pool
    let (pool_params, epoch) = generate_pool_row();
    let pools_iter = std::iter::once((pool_params.clone(), epoch));

    // DRep
    let drep_key = generate_stake_credential();
    let drep_row = generate_drep_row();

    let anchor = drep_row.anchor.clone().expect("Expected anchor to be Some");
    let deposit = drep_row.deposit;
    let registered_at = drep_row.registered_at;

    let drep_iter = std::iter::once((
        drep_key.clone(),
        (
            Resettable::Set(anchor),
            Some((deposit, registered_at)),
            registered_at.epoch(), // Match logic in `last_interaction`
        ),
    ));

    // Proposal
    /*let proposal_id = generate_proposal_id();
    let proposal_row = generate_proposal_row();
    let proposal_iter = std::iter::once((proposal_id.clone(), proposal_row.clone()));
    */

    context.save(
        &point,
        None,
        Columns {
            utxo: utxos_iter,
            pools: pools_iter,
            accounts: accounts_iter,
            dreps: drep_iter,
            cc_members: std::iter::empty(),
            proposals: std::iter::empty(),
        },
        Columns::empty(),
        std::iter::empty(),
        BTreeSet::new(),
    )?;

    Ok(SeededData {
        txin,
        output,
        account_key,
        account_row,
        pool_params,
        pool_epoch: epoch,
        drep_key,
        drep_row,
    })
}

pub fn test_read_only_store(store: &MemoryStore, seeded: SeededData) -> Result<(), StoreError> {
    // check if utxo seededData matches what is retrieved from store
    assert_eq!(store.utxo(&seeded.txin)?, Some(seeded.output.clone()));

    // check if account seededData matches what is retrieved from store
    let stored_account = store.account(&seeded.account_key)?;
    assert!(stored_account.is_some());
    let stored_account = stored_account.unwrap();
    assert_eq!(stored_account.delegatee, seeded.account_row.delegatee);
    assert_eq!(stored_account.drep, seeded.account_row.drep);
    assert_eq!(stored_account.rewards, seeded.account_row.rewards);
    assert_eq!(stored_account.deposit, seeded.account_row.deposit);

    // check if pool seededData matches what is retrieved from store
    let pool_id = seeded.pool_params.id.clone();
    let stored_pool = store.pool(&pool_id)?;
    assert!(stored_pool.is_some());
    let stored_pool = stored_pool.unwrap();
    assert_eq!(stored_pool.current_params, seeded.pool_params);
    assert_eq!(stored_pool.future_params, vec![(None, seeded.pool_epoch)]);

    // check if drep seededData matches what is retrieved from store
    let stored_drep = store
        .get_drep_for_test(&seeded.drep_key)
        .ok_or_else(|| StoreError::Internal("drep not found".into()))?;

    assert_eq!(stored_drep.anchor, seeded.drep_row.anchor);
    assert_eq!(stored_drep.deposit, seeded.drep_row.deposit);
    assert_eq!(stored_drep.registered_at, seeded.drep_row.registered_at);
    assert_eq!(
        stored_drep.last_interaction,
        seeded.drep_row.last_interaction
    );
    match (
        &stored_drep.previous_deregistration,
        &seeded.drep_row.previous_deregistration,
    ) {
        (Some(a), Some(b)) => assert_eq!(a, b),
        (None, None) => {}
        (left, right) => panic!(
            "Mismatch in previous_deregistration: left = {:?}, right = {:?}",
            left, right
        ),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::in_memory::{MemoryStore, MemoryTransactionalContext};
    use amaru_kernel::network::NetworkName;
    use amaru_kernel::EraHistory;

    #[test]
    fn test_in_memory_transaction() {
        // Get EraHistory for Preprod
        let era_history: EraHistory =
            (*Into::<&'static EraHistory>::into(NetworkName::Preprod)).clone();

        // Create a MemoryStore and transactional context
        let store = MemoryStore::new(era_history);
        let context = MemoryTransactionalContext::new(&store);

        // Seed and test
        let seeded_data = seed_store(&context).expect("seeding failed");
        test_read_only_store(&store, seeded_data).expect("read failed");
    }

    #[test]
    fn test_read_only_store_rocksdb() {
        //use amaru_stores::rocksdb::RocksDB;
        //use slot_arithmetic::testing::one_era;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("failed to create temp dir");
        //let era_history = one_era();

        //let store = RocksDB::new(temp_dir.path(), &era_history).expect("failed to create RocksDB store");

        //let context = store.transaction().expect("failed to get transaction");

        //test_read_only_store(&context).expect("rocksdb store test failed");
    }
}
