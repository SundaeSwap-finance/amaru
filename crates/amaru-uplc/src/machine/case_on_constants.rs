// Copyright 2026 PRAGMA
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

use amaru_kernel::{ProtocolVersion, protocol_version::PROTOCOL_VERSION_12};

/// Which constants a `case` term may scrutinize, selected by protocol version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CaseOnConstants {
    /// Unit, booleans, integers, lists and pairs may be scrutinized, `data` may not.
    #[default]
    NoData,
    /// A `data` built with `Constr` may be scrutinized too.
    WithData,
}

impl CaseOnConstants {
    pub fn new(protocol_version: ProtocolVersion) -> Self {
        if protocol_version >= PROTOCOL_VERSION_12 { Self::WithData } else { Self::NoData }
    }
}
