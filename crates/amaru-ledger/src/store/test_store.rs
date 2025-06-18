use std::collections::BTreeSet;

use crate::state::diff_bind::{self};
use crate::store::columns::accounts::test::any_stake_credential;
use crate::store::columns::pools::tests::any_pool_id;
use crate::store::columns::utxo::test::{any_pseudo_transaction_output, any_txin};
use crate::store::columns::{accounts, dreps, pots, proposals, slots};
use crate::store::TransactionalContext;
use crate::store::{Columns, StoreError};
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
    crate::store::dreps::tests::any_row()
        .new_tree(&mut TestRunner::default())
        .unwrap()
        .current()
}

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

fn dummy_delegatee() -> Hash<28> {
    Hash::from([0u8; 28])
}

fn dummy_drep() -> (DRep, CertificatePointer) {
    (
        DRep::Key(Hash::from([0u8; 28])),
        CertificatePointer::default(),
    )
}

fn dummy_anchor() -> Anchor {
    Anchor {
        url: "https://example.com".to_string(),
        content_hash: Hash::from([0u8; 32]),
    }
}

fn test_read_only_store<'a, C: TransactionalContext<'a>>(context: &'a C) -> Result<(), StoreError> {
    use diff_bind::Resettable;

    let point = Point::Origin;

    // Add utxo to store
    let txin = generate_txin();
    let output = generate_output();
    let utxos_iter = std::iter::once((txin, output));

    // Add account to store
    let account_stake_credential = generate_stake_credential();
    let account_row = generate_account_row();
    let accounts_iter = std::iter::once((
        account_stake_credential,
        (
            // Assume these are `Option<T>` internally, so unwrap safely or use dummy default
            Resettable::Set(account_row.delegatee.unwrap_or_else(|| dummy_delegatee())),
            Resettable::Set(account_row.drep.unwrap_or_else(|| dummy_drep())),
            Some(account_row.rewards),
            account_row.deposit,
        ),
    ));

    // Add pool to store (save takes only future params here, not Row)
    let (pool_params, epoch) = generate_pool_row();
    let pools_iter = std::iter::once((pool_params, epoch));

    // Add drep to store
    let drep_stake_credential = generate_stake_credential();
    let drep_row = generate_drep_row();
    let drep_iter = std::iter::once((
        drep_stake_credential,
        (
            Resettable::Set(drep_row.anchor.expect("Expected anchor to be Some")),
            drep_row.previous_deregistration.map(|ptr| (123u64, ptr)),
            Epoch::from(0u64),
        ),
    ));

    // Add proposal to store
    let proposal_id = generate_proposal_id();
    let proposal_row = generate_proposal_row();
    let proposal_iter = std::iter::once((proposal_id, proposal_row));

    context.save(
        &point,
        None,
        Columns {
            utxo: utxos_iter,
            pools: pools_iter,
            accounts: accounts_iter,
            dreps: drep_iter,
            cc_members: std::iter::empty(),
            proposals: proposal_iter,
        },
        Columns::empty(),
        std::iter::empty(),
        BTreeSet::new(),
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::in_memory::{MemoryStore, MemoryTransactionalContext};

    #[test]
    fn test_read_only_store_memory() {
        let store = MemoryStore::new();
        let context = MemoryTransactionalContext::new(&store);
        test_read_only_store(&context).expect("memory store test failed");
    }

    #[test]
    fn test_read_only_store_rocksdb() {
        use amaru_stores::rocksdb::{RocksDB, RocksDBTransactionalContext};
        use slot_arithmetic::testing::one_era;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("failed to create temp dir");

        let era_history = one_era();
        let store =
            RocksDB::new(temp_dir.path(), &era_history).expect("failed to create RocksDBStore");

        let context = RocksDBTransactionalContext::new(&store);
        test_read_only_store(&context).expect("rocksdb store test failed");
    }
}
