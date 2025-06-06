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

use amaru_consensus::consensus::store::ChainStore;
use amaru_kernel::{Hash, Header};
use amaru_ledger::BlockValidationResult;
use gasket::framework::*;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::trace_span;

pub type UpstreamPort = gasket::messaging::InputPort<BlockValidationResult>;

pub const EVENT_TARGET: &str = "amaru::consensus::chain_forward";

/// Forwarding stage of the consensus where blocks are stored and made
/// available to downstream peers.
///
/// TODO: currently does nothing, should store block, update chain state, and
/// forward new chain downstream

#[derive(Stage)]
#[stage(
    name = "consensus.forward",
    unit = "BlockValidationResult",
    worker = "Worker"
)]
pub struct ForwardStage {
    pub store: Arc<Mutex<dyn ChainStore<Header>>>,
    pub upstream: UpstreamPort,
}

impl ForwardStage {
    pub fn new(store: Arc<Mutex<dyn ChainStore<Header>>>) -> Self {
        Self {
            store,
            upstream: Default::default(),
        }
    }
}

pub struct Worker {}

#[async_trait::async_trait(?Send)]
impl gasket::framework::Worker<ForwardStage> for Worker {
    async fn bootstrap(_stage: &ForwardStage) -> Result<Self, WorkerError> {
        Ok(Self {})
    }

    async fn schedule(
        &mut self,
        stage: &mut ForwardStage,
    ) -> Result<WorkSchedule<BlockValidationResult>, WorkerError> {
        let unit = stage.upstream.recv().await.or_panic()?;

        Ok(WorkSchedule::Unit(unit.payload))
    }

    async fn execute(
        &mut self,
        unit: &BlockValidationResult,
        _stage: &mut ForwardStage,
    ) -> Result<(), WorkerError> {
        match unit {
            BlockValidationResult::BlockValidated(point, span) => {
                // FIXME: this span is just a placeholder to hold a link to t
                // the parent, it will be filled once we had the storage and
                // forwarding logic.
                let _span = trace_span!(
                    target: EVENT_TARGET,
                    parent: span,
                    "forward.block_validated",
                    slot = ?point.slot_or_default(),
                    hash = %Hash::<32>::from(point),
                );

                Ok(())
            }
            BlockValidationResult::BlockValidationFailed(point, span) => {
                let _span = trace_span!(
                    target: EVENT_TARGET,
                    parent: span,
                    "forward.block_validation_failed",
                    slot = ?point.slot_or_default(),
                    hash = %Hash::<32>::from(point),
                );

                Err(WorkerError::Panic)
            }
            BlockValidationResult::RolledBackTo(point, span) => {
                let _span = trace_span!(
                    target: EVENT_TARGET,
                    parent: span,
                    "rolled_back_to",
                    slot = ?point.slot_or_default(),
                    hash = %Hash::<32>::from(point),
                );

                Ok(())
            }
        }
    }
}
