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

// Expected budgets are the Plutus 1.71.0.0 results for a protocol 12 V4 cost vector of 251 entries.
use amaru_kernel::{PlutusVersion, ProtocolVersion, cost_model};
use amaru_uplc::{
    arena::Arena,
    binder::DeBruijn,
    builtin::DefaultFunction,
    constant::Integer,
    machine::{CostModel, ExBudget, MachineError, MachineVersion, RuntimeError},
    program::Program,
    syn::parse_program,
    term::Term,
};

#[expect(clippy::unwrap_used)]
fn costs() -> Vec<i64> {
    serde_json::from_str(include_str!("v4_costs.json")).unwrap()
}

#[expect(clippy::unwrap_used)]
fn evaluate(source: &str, costs: CostModel, budget: ExBudget) -> (bool, ExBudget) {
    let arena = Arena::new();
    let program = parse_program(&arena, source, ProtocolVersion::new(12, 0)).into_result().unwrap();
    let result = program.eval(&arena, costs, budget);
    (result.is_unit(), result.info.consumed_budget)
}

#[test]
fn v4_reference_budgets_and_failures() {
    for (source, expected) in [
        ("(program 1.1.0 [(lam ctx (con unit ())) (con data (I 0))])", ExBudget { cpu: 64_100, mem: 500 }),
        (include_str!("v4_grow.uplc"), ExBudget { cpu: 2_912_407_587, mem: 26_424 }),
    ] {
        assert_eq!(evaluate(source, CostModel::v4(&costs()), ExBudget::default()), (true, expected));
        assert_eq!(evaluate(source, CostModel::v4(&costs()), expected), (true, expected));
        for budget in [ExBudget { cpu: expected.cpu - 1, ..expected }, ExBudget { mem: expected.mem - 1, ..expected }] {
            assert!(!evaluate(source, CostModel::v4(&costs()), budget).0);
        }
    }
    assert!(
        !evaluate("(program 1.1.0 [(lam ctx (error)) (con data (I 0))])", CostModel::v4(&costs()), ExBudget::default())
            .0
    );
}

#[test]
fn v4_arithmetic_operands_are_unbounded_but_v3_is_unchanged() {
    for builtin in [
        DefaultFunction::AddInteger,
        DefaultFunction::SubtractInteger,
        DefaultFunction::MultiplyInteger,
        DefaultFunction::DivideInteger,
        DefaultFunction::QuotientInteger,
        DefaultFunction::RemainderInteger,
        DefaultFunction::ModInteger,
        DefaultFunction::LessThanInteger,
        DefaultFunction::LessThanEqualsInteger,
    ] {
        for negative in [false, true] {
            let huge = Integer::from(1) << 262_144usize;
            let huge = if negative { -huge } else { huge };
            for reverse in [false, true] {
                let arena = Arena::new();
                let large = Term::integer(&arena, &huge);
                let small = Term::integer_from(&arena, 1);
                let (x, y) = if reverse { (small, large) } else { (large, small) };
                let term = Term::<DeBruijn>::builtin(&arena, builtin).apply(&arena, x).apply(&arena, y);
                let program = Program::new(&arena, MachineVersion::V1_1_0, term);
                let result = program.eval(&arena, CostModel::v4(&costs()), ExBudget::max());
                assert!(result.is_success(), "{builtin:?}: {result:?}");
                let old = program.eval(
                    &arena,
                    CostModel::new(PlutusVersion::V3, ProtocolVersion::new(12, 0), &costs()),
                    ExBudget::max(),
                );
                assert!(matches!(old.error(), Some(MachineError::Runtime(RuntimeError::IntegerOutOfBounds(_)))));
            }
        }
    }
}

#[test]
fn v4_keeps_bytestring_and_cons_bounds() {
    for length in [65_536, 65_537] {
        let arena = Arena::new();
        let bytes = vec![0; length];
        let term = Term::<DeBruijn>::builtin(&arena, DefaultFunction::Blake2b_256)
            .apply(&arena, Term::byte_string(&arena, &bytes));
        let program = Program::new(&arena, MachineVersion::V1_1_0, term);
        assert_eq!(program.eval(&arena, CostModel::v4(&costs()), ExBudget::max()).is_success(), length == 65_536);
    }
    for byte in [-1, 0, 255, 256] {
        let arena = Arena::new();
        let term = Term::<DeBruijn>::builtin(&arena, DefaultFunction::ConsByteString)
            .apply(&arena, Term::integer_from(&arena, byte))
            .apply(&arena, Term::byte_string(&arena, &[]));
        let program = Program::new(&arena, MachineVersion::V1_1_0, term);
        assert_eq!(
            program.eval(&arena, CostModel::v4(&costs()), ExBudget::max()).is_success(),
            (0..=255).contains(&byte)
        );
    }
}

#[test]
fn v4_bounds_shift_amounts_to_int64() {
    for (amount, in_range) in [("9223372036854775807", true), ("9223372036854775808", false)] {
        let source = format!("(program 1.1.0 [(builtin shiftByteString) (con bytestring #00) (con integer {amount})])");
        let arena = Arena::new();
        let program = parse_program(&arena, &source, ProtocolVersion::new(12, 0)).into_result().unwrap();
        let result = program.eval(&arena, CostModel::v4(&costs()), ExBudget::max());
        if in_range {
            assert!(matches!(result.error(), Some(MachineError::OutOfExError(_))), "{amount}: {result:?}");
        } else {
            assert!(
                matches!(result.error(), Some(MachineError::Runtime(RuntimeError::OutsideUsizeBounds(_)))),
                "{amount}: {result:?}"
            );
        }
    }
}

#[test]
fn v4_bounds_write_bits_input_length() {
    for length in [4_096, 4_097] {
        let source = "(program 1.1.0 (lam x [(builtin writeBits) x (con (list integer) []) (con bool True)]))";
        let arena = Arena::new();
        let bytes = vec![0; length];
        let program = parse_program(&arena, source, ProtocolVersion::new(12, 0)).into_result().unwrap();
        let program = program.apply(&arena, Term::byte_string(&arena, &bytes));
        let result = program.eval(&arena, CostModel::v4(&cost_model::DEFAULT_COST_MODEL_E_V3), ExBudget::max());
        if length == 4_096 {
            assert!(result.is_success(), "{length}: {result:?}");
        } else {
            assert!(
                matches!(
                    result.error(),
                    Some(MachineError::Runtime(RuntimeError::WriteBitsInputTooLong(4_097, 4_096)))
                ),
                "{length}: {result:?}"
            );
        }
    }
}

#[test]
fn v4_missing_costs_are_expensive_and_future_tail_does_not_change_unit() {
    let source = "(program 1.1.0 [(lam ctx (con unit ())) (con data (I 0))])";
    assert!(!evaluate(source, CostModel::v4(&[]), ExBudget::default()).0);
    let short = costs();
    let mut extended = short.clone();
    extended.resize(369, i64::MAX);
    assert_eq!(
        evaluate(source, CostModel::v4(&short), ExBudget::default()),
        evaluate(source, CostModel::v4(&extended), ExBudget::default())
    );
}

#[test]
fn v4_uses_the_supplied_cost_coefficients() {
    let source = "(program 1.1.0 [(lam ctx (con unit ())) (con data (I 0))])";
    let mut supplied = costs();
    supplied[29] += 1_000; // CekStartupCost-exBudgetCPU.
    assert_eq!(
        evaluate(source, CostModel::v4(&supplied), ExBudget::default()),
        (true, ExBudget { cpu: 65_100, mem: 500 })
    );
}

#[test]
fn v4_unpriced_insert_coin_exhausts_budget() {
    let source = "(program 1.1.0 [(lam ctx [(lam ignored (con unit ())) [(builtin insertCoin) \
        (con bytestring #) (con bytestring #) (con integer 1) (con value [(#, [(#, 1)])])]]) \
        (con data (I 0))])";
    let arena = Arena::new();
    let program = parse_program(&arena, source, ProtocolVersion::new(12, 0)).into_result().unwrap();
    let result = program.eval(&arena, CostModel::v4(&costs()), ExBudget::default());
    assert!(matches!(result.error(), Some(MachineError::OutOfExError(_))), "{result:?}");
}

#[test]
fn v4_retains_utf8_byte_costing() {
    let source = r#"(program 1.1.0 [(lam ctx [(lam ignored (con unit ()))
        [(builtin appendString) (con string "é") (con string "ß")]]) (con data (I 0))])"#;
    assert_eq!(
        evaluate(source, CostModel::v4(&costs()), ExBudget::default()),
        (true, ExBudget { cpu: 177_100, mem: 1_204 })
    );
}
