use std::collections::BTreeSet;

use crate::state::diff_bind::{self};
use crate::store::columns::accounts::test::any_stake_credential;
use crate::store::columns::utxo::test::{any_pseudo_transaction_output, any_txin};
use crate::store::columns::{accounts, cc_members, dreps};
use crate::store::in_memory::MemoryStore;
use crate::store::{Columns, StoreError};
use crate::store::{ReadOnlyStore, TransactionalContext};
use amaru_kernel::TransactionOutput;
use amaru_kernel::{Anchor, Hash, Point, PoolParams, Slot, StakeCredential, TransactionInput};
use amaru_kernel::{EraHistory, PoolId};
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

fn generate_pool_row() -> (PoolParams, Epoch) {
    let pool_params = crate::store::pools::tests::any_pool_params()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current();

    let epoch = Epoch::from(0u64);

    (pool_params, epoch)
}

fn generate_pool_id() -> PoolId {
    crate::store::pools::tests::any_pool_id()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current()
}

/*
fn generate_pots_row() -> pots::Row {
    crate::store::pots::test::any_row()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current()
}*/

fn generate_slot() -> Slot {
    crate::store::slots::tests::any_slot()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current()
}

fn generate_new_drep_row() -> dreps::Row {
    let mut row = crate::store::dreps::tests::any_row()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current();
    if row.anchor.is_none() {
        row.anchor = Some(dummy_anchor());
    }
    row.previous_deregistration = None;
    row.last_interaction = None;

    row
}

fn dummy_anchor() -> Anchor {
    Anchor {
        url: "https://example.com".to_string(),
        content_hash: Hash::from([0u8; 32]),
    }
}

/*
fn generate_proposal_id() -> ProposalId {
    crate::store::proposals::tests::any_proposal_id()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current()
}

fn generate_proposal_row() -> proposals::Row {
    crate::store::proposals::tests::any_row()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current()
}
*/

fn generate_cc_member_row() -> cc_members::Row {
    let mut runner = TestRunner::default();

    let mut row = crate::store::cc_members::test::any_row()
        .new_tree(&mut runner)
        .unwrap()
        .current();

    if row.hot_credential.is_none() {
        let hot = crate::store::cc_members::test::any_stake_credential()
            .new_tree(&mut runner)
            .unwrap()
            .current();
        row.hot_credential = Some(hot);
    }

    row
}

#[derive(Debug, Clone)]
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
    //pub cc_member_key: StakeCredential,
    //pub cc_member_row: cc_members::Row,
    //pub slot_leader: PoolId,
    //pub point: Point,
}

pub fn add_test_data_to_store<'a, C: TransactionalContext<'a>>(
    context: &'a C,
    era_history: &EraHistory,
) -> Result<SeededData, StoreError> {
    use diff_bind::Resettable;

    // utxos
    let txin = generate_txin();
    let output = generate_output();
    let utxos_iter = std::iter::once((txin.clone(), output.clone()));

    // accounts
    let account_key = generate_stake_credential();
    let account_key_clone = account_key.clone(); // clone BEFORE it gets moved

    let account_row = generate_account_row();

    let delegatee = match &account_row.delegatee {
        Some(pool_id) => Resettable::Set(pool_id.clone()),
        None => Resettable::Reset,
    };

    let drep = match &account_row.drep {
        Some(drep_pair) => Resettable::Set(drep_pair.clone()),
        None => Resettable::Reset,
    };

    let rewards = Some(account_row.rewards);
    let deposit = account_row.deposit;

    let accounts_iter = std::iter::once((account_key_clone, (delegatee, drep, rewards, deposit)));

    // pools
    let (pool_params, epoch) = generate_pool_row();
    let pools_iter = std::iter::once((pool_params.clone(), epoch));

    // dreps
    let drep_key = generate_stake_credential();
    let drep_row = generate_new_drep_row();

    let anchor = drep_row.anchor.clone().expect("Expected anchor to be Some");
    let deposit = drep_row.deposit;
    let registered_at = drep_row.registered_at;

    let epoch = era_history
        .slot_to_epoch(registered_at.transaction.slot)
        .expect("Failed to convert slot to epoch");

    let drep_iter = std::iter::once((
        drep_key.clone(),
        (
            Resettable::Set(anchor),
            Some((deposit, registered_at)),
            epoch,
        ),
    ));
    // cc_members
    let cc_member_key = generate_stake_credential();
    let cc_member_row = generate_cc_member_row();
    let hot_credential = cc_member_row
        .hot_credential
        .clone()
        .expect("Expected Some hot_credential");
    let cc_members_iter = std::iter::once((
        cc_member_key.clone(),
        Resettable::Set(hot_credential.clone()),
    ));

    /*
    // proposals
    let proposal_id = generate_proposal_id();
    let proposal_row = generate_proposal_row();
    let proposal_iter = std::iter::once((proposal_id.clone(), proposal_row.clone()));
    */

    let slot = generate_slot();
    let point = Point::Specific(slot.into(), Hash::from([0u8; 32]).to_vec());
    let slot_leader = generate_pool_id();
    context.save(
        &point,
        Some(&slot_leader),
        Columns {
            utxo: utxos_iter,
            pools: pools_iter,
            accounts: accounts_iter,
            dreps: drep_iter,
            cc_members: cc_members_iter,
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
        /*
        proposal_id.      // Need Ord implemented on ProposalId
        proposal_row      // ^^^
        cc_member_key,    // Need cc_members to be exposed via iterator
        cc_member_row,    // ^^^
        slot_leader,      // Need slots to be exposed via iterator
        point,            // ^^^
        */
    })
}

pub fn test_read_utxo(store: &impl ReadOnlyStore, seeded: &SeededData) {
    let result = store
        .utxo(&seeded.txin)
        .expect("failed to read UTXO from store");

    assert_eq!(
        result,
        Some(seeded.output.clone()),
        "UTXO did not match seeded output"
    );
}

pub fn test_read_account(store: &impl ReadOnlyStore, seeded: &SeededData) {
    let stored_account = store
        .account(&seeded.account_key)
        .expect("failed to read account from store");

    assert!(
        stored_account.is_some(),
        "account not found in store for seeded key"
    );

    let stored_account = stored_account.unwrap();

    assert_eq!(
        stored_account.delegatee, seeded.account_row.delegatee,
        "delegatee mismatch"
    );
    assert_eq!(
        stored_account.drep, seeded.account_row.drep,
        "drep mismatch"
    );
    assert_eq!(
        stored_account.rewards, seeded.account_row.rewards,
        "rewards mismatch"
    );
    assert_eq!(
        stored_account.deposit, seeded.account_row.deposit,
        "deposit mismatch"
    );
}

pub fn test_read_pool(store: &impl ReadOnlyStore, seeded: &SeededData) {
    let pool_id = seeded.pool_params.id.clone();
    let stored_pool = store
        .pool(&pool_id)
        .expect("failed to read pool from store");

    assert!(
        stored_pool.is_some(),
        "pool not found in store for seeded id"
    );

    let stored_pool = stored_pool.unwrap();

    assert_eq!(
        stored_pool.current_params, seeded.pool_params,
        "current pool params mismatch"
    );
    assert_eq!(
        stored_pool.future_params,
        vec![(None, seeded.pool_epoch)],
        "future pool params mismatch"
    );
}

pub fn test_read_drep(store: &MemoryStore, seeded: &SeededData) {
    let stored_drep = store
        .get_drep_for_test(&seeded.drep_key)
        .expect("drep not found in store");

    assert_eq!(
        stored_drep.anchor, seeded.drep_row.anchor,
        "drep anchor mismatch"
    );
    assert_eq!(
        stored_drep.deposit, seeded.drep_row.deposit,
        "drep deposit mismatch"
    );
    assert_eq!(
        stored_drep.registered_at, seeded.drep_row.registered_at,
        "drep registration time mismatch"
    );
    assert_eq!(
        stored_drep.last_interaction, seeded.drep_row.last_interaction,
        "drep last interaction mismatch"
    );

    match (
        &stored_drep.previous_deregistration,
        &seeded.drep_row.previous_deregistration,
    ) {
        (Some(a), Some(b)) => assert_eq!(a, b, "drep previous deregistration mismatch"),
        (None, None) => {}
        (left, right) => panic!(
            "Mismatch in previous_deregistration: left = {:?}, right = {:?}",
            left, right
        ),
    }
}

/* Disabled until ReadOnlyStore getter is implemented for cc_members column
pub fn test_read_cc_member(store: &MemoryStore, seeded: &SeededData) {
    assert_eq!(
        store.cc_member(&seeded.cc_member_key),
        Some(seeded.cc_member_row.clone()),
        "cc_member mismatch"
    );
}*/

// TODO: Implement Ord on ProposalId in pallas-primitives to allow proposals to be stored in memory

pub fn test_remove_utxo<'a>(
    context: &'a impl TransactionalContext<'a>,
    store: &impl ReadOnlyStore,
    seeded: &SeededData,
) {
    let point = Point::Origin;

    let remove = Columns {
        utxo: std::iter::once(seeded.txin.clone()),
        pools: std::iter::empty(),
        accounts: std::iter::empty(),
        dreps: std::iter::empty(),
        cc_members: std::iter::empty(),
        proposals: std::iter::empty(),
    };

    context
        .save(
            &point,
            None,
            Columns::empty(),
            remove,
            std::iter::empty(),
            BTreeSet::new(),
        )
        .expect("utxo removal failed");

    assert_eq!(
        store.utxo(&seeded.txin).expect("utxo lookup failed"),
        None,
        "utxo was not properly removed"
    );
}

fn test_remove_account<'a>(
    context: &'a impl TransactionalContext<'a>,
    store: &impl ReadOnlyStore,
    seeded: &SeededData,
) -> Result<(), StoreError> {
    let point = Point::Origin;

    let remove = Columns {
        utxo: std::iter::empty(),
        pools: std::iter::empty(),
        accounts: std::iter::once(seeded.account_key.clone()),
        dreps: std::iter::empty(),
        cc_members: std::iter::empty(),
        proposals: std::iter::empty(),
    };

    context.save(
        &point,
        None,
        Columns::empty(),
        remove,
        std::iter::empty(),
        BTreeSet::new(),
    )?;

    assert_eq!(store.account(&seeded.account_key)?, None);

    Ok(())
}

fn test_remove_pool<'a>(
    context: &'a impl TransactionalContext<'a>,
    store: &impl ReadOnlyStore,
    seeded: &SeededData,
) -> Result<(), StoreError> {
    let point = Point::Origin;

    let remove = Columns {
        utxo: std::iter::empty(),
        pools: std::iter::once((seeded.pool_params.id.clone(), seeded.pool_epoch)),
        accounts: std::iter::empty(),
        dreps: std::iter::empty(),
        cc_members: std::iter::empty(),
        proposals: std::iter::empty(),
    };

    context.save(
        &point,
        None,
        Columns::empty(),
        remove,
        std::iter::empty(),
        BTreeSet::new(),
    )?;

    assert_eq!(store.pool(&seeded.pool_params.id)?, None);

    Ok(())
}

fn test_remove_drep<'a>(
    context: &'a impl TransactionalContext<'a>,
    store: &impl ReadOnlyStore,
    seeded: &SeededData,
) -> Result<(), StoreError> {
    let point = Point::Origin;

    let drep_registered_at = store
        .iter_dreps()?
        .find(|(key, _)| *key == seeded.drep_key)
        .and_then(|(_, row)| Some(row.registered_at))
        .ok_or_else(|| StoreError::Internal("DRep not found before removal".into()))?;

    let remove = Columns {
        utxo: std::iter::empty(),
        pools: std::iter::empty(),
        accounts: std::iter::empty(),
        dreps: std::iter::once((seeded.drep_key.clone(), drep_registered_at)),
        cc_members: std::iter::empty(),
        proposals: std::iter::empty(),
    };

    assert!(
        store.iter_dreps()?.any(|(key, _)| key == seeded.drep_key),
        "DRep not present before removal"
    );

    context.save(
        &point,
        None,
        Columns::empty(),
        remove,
        std::iter::empty(),
        BTreeSet::new(),
    )?;

    let drep_exists = store.iter_dreps()?.any(|(key, _)| key == seeded.drep_key);

    assert!(!drep_exists, "DRep was not removed");

    Ok(())
}

fn test_refund_account<'a>(
    context: &'a impl TransactionalContext<'a>,
    seeded: &SeededData,
) -> Result<(), StoreError> {
    // --- Existing account ---
    let refund_amount = 100;
    let deposit_before = seeded.account_row.deposit;

    let unrefunded = context.refund(&seeded.account_key, refund_amount)?;
    assert_eq!(unrefunded, 0, "Refund to existing account should succeed");

    let mut deposit_after = None;
    context.with_accounts(|mut accounts| {
        deposit_after = accounts
            .find(|(key, _)| *key == seeded.account_key)
            .and_then(|(_, row)| row.borrow().as_ref().map(|acc| acc.deposit));
    })?;

    let deposit_after =
        deposit_after.ok_or_else(|| StoreError::Internal("Missing account after refund".into()))?;

    assert_eq!(
        deposit_after,
        deposit_before + refund_amount,
        "Deposit should increase by refund amount"
    );

    // --- Missing account ---
    let unknown = generate_stake_credential();
    assert_ne!(unknown, seeded.account_key);

    let refund_amount = 123;
    let unrefunded = context.refund(&unknown, refund_amount)?;
    assert_eq!(
        unrefunded, refund_amount,
        "Missing account should return full refund amount"
    );

    Ok(())
}

fn test_epoch_transition<'a>(context: &'a impl TransactionalContext<'a>) -> Result<(), StoreError> {
    use crate::store::EpochTransitionProgress;

    let from = None;
    let to = Some(EpochTransitionProgress::EpochEnded);

    let success = context.try_epoch_transition(from.clone(), to.clone())?;
    assert!(
        success,
        "Expected epoch transition to succeed when previous state matches"
    );

    let repeat = context.try_epoch_transition(from, to)?;
    assert!(
        !repeat,
        "Expected second transition from outdated state to fail"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::in_memory::{MemoryStore, MemoryTransactionalContext};
    use amaru_kernel::network::NetworkName;
    use amaru_kernel::EraHistory;

    #[test]
    fn test_in_memory_store() -> Result<(), StoreError> {
        let era_history: EraHistory =
            (*Into::<&'static EraHistory>::into(NetworkName::Preprod)).clone();
        let store = MemoryStore::new(era_history.clone());
        let context = MemoryTransactionalContext::new(&store);

        let seeded =
            add_test_data_to_store(&context, &era_history).expect("adding data to store failed");

        // Verify seeded data can be read back correctly
        test_read_utxo(&store, &seeded);
        test_read_account(&store, &seeded);
        test_read_pool(&store, &seeded);
        test_read_drep(&store, &seeded);
        //test_read_cc_member(&store, &seeded);
        //test_read_proposal(&store, &seeded);

        // Verify store updates through context
        test_refund_account(&context, &seeded)?;
        test_epoch_transition(&context)?;
        //test_slot_updated(&store, &seeded);

        // Verify removal of seeded data
        test_remove_utxo(&context, &store, &seeded);
        test_remove_account(&context, &store, &seeded)?;
        test_remove_pool(&context, &store, &seeded)?;
        test_remove_drep(&context, &store, &seeded)?;
        //test_remove_cc_member(&context, &store, &seeded);
        //test_remove_proposal(&context, &store, &seeded);

        Ok(())
    }

    #[test]
    fn test_rocksdb_store() -> Result<(), StoreError> {
        /*
        use crate::store::{ReadOnlyStore, Store, TransactionalContext};
        use amaru_stores::rocksdb::RocksDB;
        use tempfile::TempDir;

        let era_history: EraHistory =
            (*Into::<&'static EraHistory>::into(NetworkName::Preprod)).clone();

        let tmp_dir = TempDir::new().expect("failed to create temp dir");

        let store: RocksDB = RocksDB::new(tmp_dir.path(), &era_history)
            .map_err(|e| StoreError::Internal(e.into()))?;

        let context = <RocksDB as Store>::create_transaction(&store);

        let seeded =
            add_test_data_to_store(&context, &era_history).expect("adding data to store failed");

        // Verify seeded data can be read back correctly
        test_read_utxo(&store, &seeded);
        test_read_account(&store, &seeded);
        test_read_pool(&store, &seeded);
        test_read_drep(&store, &seeded);
        //test_read_cc_member(&store, &seeded);
        //test_read_proposal(&store, &seeded);

        // Verify store updates through context
        test_refund_account(&context, &seeded)?;
        test_epoch_transition(&context)?;
        //test_slot_updated(&store, &seeded);

        // Verify removal of seeded data
        test_remove_utxo(&context, &store, &seeded);
        test_remove_account(&context, &store, &seeded)?;
        test_remove_pool(&context, &store, &seeded)?;
        test_remove_drep(&context, &store, &seeded)?;
        //test_remove_cc_member(&context, &store, &seeded);
        //test_remove_proposal(&context, &store, &seeded);
        */
        Ok(())
    }
}
