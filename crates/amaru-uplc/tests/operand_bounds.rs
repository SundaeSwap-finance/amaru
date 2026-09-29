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

use amaru_kernel::{PlutusVersion, ProtocolVersion};
use amaru_uplc::{
    arena::Arena,
    binder::DeBruijn,
    builtin::DefaultFunction,
    machine::{CostModel, ExBudget, MachineError, MachineVersion, RuntimeError},
    program::Program,
    syn::parse_program,
    term::Term,
};

// The bound applies to CByteString operands, not results or every bytestring argument.
// Plutus 1.68.0.0, Default/Builtins.hs and Default/Universe/Cardano.hs:
// https://github.com/IntersectMBO/plutus/tree/9e17e2404dc6988c908b1fea099dde202df73b6a
#[test]
fn hash_operand_limit_depends_on_protocol() {
    for language in [PlutusVersion::V1, PlutusVersion::V2, PlutusVersion::V3] {
        for protocol in [10, 11, 12] {
            for length in [65_535, 65_536, 65_537, 131_072] {
                let arena = Arena::new();
                let bytes = vec![0; length];
                let term = Term::<DeBruijn>::builtin(&arena, DefaultFunction::Blake2b_256)
                    .apply(&arena, Term::byte_string(&arena, &bytes));
                let program = Program::new(&arena, MachineVersion::V1_0_0, term);
                let costs = CostModel::default_for(language, ProtocolVersion::new(protocol, 0));
                let result = program.eval(&arena, costs, ExBudget::max());
                assert_eq!(
                    result.is_success(),
                    protocol < 11 || length <= 65_536,
                    "{language:?}, protocol {protocol}, length {length}: {result:?}"
                );
            }
        }
    }
}

#[test]
fn append_can_produce_oversized_result_but_hash_cannot_consume_it() {
    for hash in [false, true] {
        let arena = Arena::new();
        let bytes = vec![0; 65_536];
        let arg = Term::byte_string(&arena, &bytes);
        let append =
            Term::<DeBruijn>::builtin(&arena, DefaultFunction::AppendByteString).apply(&arena, arg).apply(&arena, arg);
        let term =
            if hash { Term::builtin(&arena, DefaultFunction::Blake2b_256).apply(&arena, append) } else { append };
        let program = Program::new(&arena, MachineVersion::V1_1_0, term);
        assert_eq!(program.eval(&arena, CostModel::v3(), ExBudget::max()).is_success(), !hash);
    }
}

#[test]
fn unrestricted_bytestring_consumers_accept_oversized_values() {
    for builtin in [DefaultFunction::BData, DefaultFunction::LengthOfByteString] {
        let arena = Arena::new();
        let bytes = vec![0; 131_072];
        let term = Term::<DeBruijn>::builtin(&arena, builtin).apply(&arena, Term::byte_string(&arena, &bytes));
        let program = Program::new(&arena, MachineVersion::V1_1_0, term);
        assert!(program.eval(&arena, CostModel::v3(), ExBudget::max()).is_success());
    }
}

// Each position below is a CByteString argument in the ledger builtin signature.
// Other arguments stay small so failure identifies the selected operand.
#[test]
fn selected_bytestring_arguments_are_bounded() {
    let applications = [
        "appendByteString x (con bytestring #)",
        "appendByteString (con bytestring #) x",
        "consByteString (con integer 0) x",
        "sliceByteString (con integer 0) (con integer 1) x",
        "indexByteString x (con integer 0)",
        "equalsByteString x (con bytestring #)",
        "equalsByteString (con bytestring #) x",
        "lessThanByteString x (con bytestring #)",
        "lessThanByteString (con bytestring #) x",
        "lessThanEqualsByteString x (con bytestring #)",
        "lessThanEqualsByteString (con bytestring #) x",
        "sha2_256 x",
        "sha3_256 x",
        "blake2b_256 x",
        "verifyEd25519Signature (con bytestring #) x (con bytestring #)",
        "verifySchnorrSecp256k1Signature (con bytestring #) x (con bytestring #)",
        "decodeUtf8 x",
        "bls12_381_G1_hashToGroup x (con bytestring #)",
        "bls12_381_G2_hashToGroup x (con bytestring #)",
        "keccak_256 x",
        "blake2b_224 x",
        "byteStringToInteger (con bool True) x",
        "andByteString (con bool True) x (con bytestring #)",
        "andByteString (con bool True) (con bytestring #) x",
        "orByteString (con bool True) x (con bytestring #)",
        "orByteString (con bool True) (con bytestring #) x",
        "xorByteString (con bool True) x (con bytestring #)",
        "xorByteString (con bool True) (con bytestring #) x",
        "complementByteString x",
        "readBit x (con integer 0)",
        "countSetBits x",
        "findFirstSetBit x",
        "ripemd_160 x",
    ];
    for application in applications {
        let (builtin, arguments) = application.split_once(' ').unwrap();
        let source = format!("(program 1.1.0 (lam x [(builtin {builtin}) {arguments}]))");
        let arena = Arena::new();
        let bytes = vec![0; 65_537];
        let program = parse_program(&arena, &source, ProtocolVersion::new(11, 0)).into_result().unwrap();
        let program = program.apply(&arena, Term::byte_string(&arena, &bytes));
        let result = program.eval(&arena, CostModel::v3(), ExBudget::max());
        assert!(
            matches!(result.error(), Some(MachineError::Runtime(RuntimeError::ByteStringOperandTooLarge(65_537)))),
            "{application}: {result:?}"
        );
    }
}

#[test]
fn valid_boundary_hash_still_obeys_execution_budget() {
    let arena = Arena::new();
    let bytes = vec![0; 65_536];
    let term = Term::<DeBruijn>::builtin(&arena, DefaultFunction::Blake2b_256)
        .apply(&arena, Term::byte_string(&arena, &bytes));
    let program = Program::new(&arena, MachineVersion::V1_1_0, term);
    let result = program.eval(&arena, CostModel::v3(), ExBudget::max());
    assert!(result.is_success());
    let budget = result.info.consumed_budget;
    assert!(program.eval(&arena, CostModel::v3(), budget).is_success());
    for insufficient in [ExBudget::new(budget.mem - 1, budget.cpu), ExBudget::new(budget.mem, budget.cpu - 1)] {
        let result = program.eval(&arena, CostModel::v3(), insufficient);
        assert!(matches!(result.error(), Some(MachineError::OutOfExError(_))));
    }
}
