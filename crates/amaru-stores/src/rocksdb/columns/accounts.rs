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

use crate::rocksdb::common::{as_key, as_value, PREFIX_LEN};
use amaru_kernel::Lovelace;
use amaru_ledger::store::StoreError;
use rocksdb::Transaction;
use tracing::{error, info};

use amaru_ledger::store::columns::accounts::{Key, Row, Value, EVENT_TARGET};

/// Name prefixed used for storing Account entries. UTF-8 encoding for "acct"
pub const PREFIX: [u8; PREFIX_LEN] = [0x61, 0x63, 0x63, 0x74];

/// Register a new credential, with or without a stake pool.
pub fn add<DB>(
    db: &Transaction<'_, DB>,
    rows: impl Iterator<Item = (Key, Value)>,
) -> Result<(), StoreError> {
    for (credential, (delegatee, drep, deposit, rewards)) in rows {
        let key = as_key(&PREFIX, &credential);

        // In case where a registration already exists, then we must only update the underlying
        // entry, while preserving the reward amount.
        if let Some(mut row) = db
            .get(&key)
            .map_err(|err| StoreError::Internal(err.into()))?
            .map(Row::unsafe_decode)
        {
            delegatee.set_or_reset(&mut row.delegatee);
            drep.set_or_reset(&mut row.drep);

            if let Some(deposit) = deposit {
                row.deposit = deposit;
            }

            db.put(key, as_value(row))
                .map_err(|err| StoreError::Internal(err.into()))?;
        } else if let Some(deposit) = deposit {
            let mut row = Row {
                deposit,
                delegatee: None,
                drep: None,
                rewards,
            };

            delegatee.set_or_reset(&mut row.delegatee);
            drep.set_or_reset(&mut row.drep);

            db.put(key, as_value(row))
                .map_err(|err| StoreError::Internal(err.into()))?;
        } else {
            error!(
                target: EVENT_TARGET,
                ?credential,
                "add.register_no_deposit",
            )
        };
    }

    Ok(())
}

/// Reset rewards counter of many accounts.
pub fn reset_many<DB>(
    db: &Transaction<'_, DB>,
    rows: impl Iterator<Item = Key>,
) -> Result<(), StoreError> {
    for credential in rows {
        let key = as_key(&PREFIX, &credential);

        if let Some(mut row) = db
            .get(&key)
            .map_err(|err| StoreError::Internal(err.into()))?
            .map(Row::unsafe_decode)
        {
            row.rewards = 0;
            db.put(key, as_value(row))
                .map_err(|err| StoreError::Internal(err.into()))?;
        } else {
            error!(
                target: EVENT_TARGET,
                ?credential,
                "reset.no_account",
            )
        }
    }

    Ok(())
}

/// Alter balance of a specific account. If the account did not exist, returns the leftovers
/// amount that couldn't be allocated to the account.
pub fn set<DB>(
    db: &Transaction<'_, DB>,
    credential: Key,
    with_rewards: impl FnOnce(Lovelace) -> Lovelace,
) -> Result<Lovelace, StoreError> {
    let key = as_key(&PREFIX, &credential);

    if let Some(mut row) = db
        .get(&key)
        .map_err(|err| StoreError::Internal(err.into()))?
        .map(Row::unsafe_decode)
    {
        row.rewards = with_rewards(row.rewards);
        db.put(key, as_value(row))
            .map_err(|err| StoreError::Internal(err.into()))?;
        return Ok(0);
    }

    info!(
        target: EVENT_TARGET,
        ?credential,
        "set.no_account",
    );

    Ok(with_rewards(0))
}

/// Clear a stake credential registration.
pub fn remove<DB>(
    db: &Transaction<'_, DB>,
    rows: impl Iterator<Item = Key>,
) -> Result<(), StoreError> {
    for credential in rows {
        db.delete(as_key(&PREFIX, &credential))
            .map_err(|err| StoreError::Internal(err.into()))?;
    }

    Ok(())
}
