// Copyright 2025 PRAGMA
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

use crate::{
    context::{AccountsSlice, WitnessSlice},
    rules::TransactionField,
};
use amaru_kernel::{Address, HasOwnership, Lovelace, RewardAccount};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum InvalidWithdrawals {
    #[error("unexpected bytes instead of reward account in {context:?} at position {position}")]
    MalformedRewardAccount {
        bytes: Vec<u8>,
        context: TransactionField,
        position: usize,
    },
}

pub(crate) fn execute<C>(
    context: &mut C,
    withdrawals: Option<&Vec<(RewardAccount, Lovelace)>>,
) -> Result<(), InvalidWithdrawals>
where
    C: WitnessSlice + AccountsSlice,
{
    if let Some(withdrawals) = withdrawals {
        withdrawals
            .iter()
            .enumerate()
            .try_for_each(|(position, (raw_account, _))| {
                // TODO: This parsing should happen when we first deserialise the block, and
                // not in the middle of rules validations.
                let credential = Address::from_bytes(raw_account)
                    .ok()
                    .and_then(|account| account.credential())
                    .ok_or_else(|| InvalidWithdrawals::MalformedRewardAccount {
                        bytes: raw_account.to_vec(),
                        context: TransactionField::Withdrawals,
                        position,
                    })?;

                context.require_witness(credential.clone());

                context.withdraw_from(credential);

                Ok(())
            })?;
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use amaru_kernel::{cbor, KeepRaw, MintedTransactionBody};
    use test_case::test_case;

    use crate::{
        context::assert::{AssertPreparationContext, AssertValidationContext},
        tests::{include_expected_traces, include_transaction_body, verify_traces, with_tracing},
    };

    static TEST_CASES: LazyLock<[(KeepRaw<'_, MintedTransactionBody<'_>>, &str); 2]> =
        LazyLock::new(|| {
            macro_rules! include_testing_data {
                ($path:literal, $hash:literal) => {
                    (
                        include_transaction_body!($path, $hash),
                        include_expected_traces!($path, $hash),
                    )
                };
            }

            [
                include_testing_data!(
                    "../../../tests",
                    "f861e92f12e12a744e1392a29fee5c49b987eae5e75c805f14e6ecff4ef13ff7"
                ),
                include_testing_data!(
                    "../../../tests",
                    "a81147b58650b80f08986b29dad7f5efedd53ff215c17659f9dd0596e9a3d227"
                ),
            ]
        });

    #[test_case(&TEST_CASES[0])]
    #[test_case(&TEST_CASES[1])]
    fn valid_withdrawal((tx, expected_traces): &(KeepRaw<'_, MintedTransactionBody<'_>>, &str)) {
        with_tracing(|collector| {
            let mut context = AssertValidationContext::from(AssertPreparationContext {
                utxo: Default::default(),
            });

            assert!(super::execute(&mut context, tx.withdrawals.as_deref()).is_ok());

            match verify_traces(collector.lines.lock().unwrap().clone(), expected_traces) {
                Ok(_) => {}
                Err(e) => panic!("{:?}", e),
            }
        });
    }
}
