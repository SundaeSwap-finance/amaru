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

use amaru_kernel::{
    PlutusVersion,
    cost_model::DEFAULT_COST_MODEL_E_V3,
    protocol_version::{PROTOCOL_VERSION_11, PROTOCOL_VERSION_12},
};
use amaru_uplc::{
    arena::Arena,
    binder::DeBruijn,
    data::PlutusData,
    flat,
    machine::{CostModel, EvalResult, ExBudget, MachineError},
    program::Program,
    syn::parse_program,
    term::Term,
};

#[derive(Debug, PartialEq)]
enum Outcome {
    Unit,
    NonConstrScrutinized,
    MissingCaseBranch,
    Other(String),
}

fn outcome(result: &EvalResult<'_, DeBruijn>) -> Outcome {
    match result.error() {
        None if result.is_unit() => Outcome::Unit,
        None => Outcome::Other(format!("{result:?}")),
        Some(MachineError::NonConstrScrutinized(_)) => Outcome::NonConstrScrutinized,
        Some(MachineError::MissingCaseBranch(..)) => Outcome::MissingCaseBranch,
        Some(error) => Outcome::Other(error.to_string()),
    }
}

fn v4() -> CostModel {
    CostModel::v4(&DEFAULT_COST_MODEL_E_V3)
}

fn v3_at_protocol_12() -> CostModel {
    CostModel::new(PlutusVersion::V3, PROTOCOL_VERSION_12, &DEFAULT_COST_MODEL_E_V3)
}

fn v3_at_protocol_11() -> CostModel {
    CostModel::new(PlutusVersion::V3, PROTOCOL_VERSION_11, &DEFAULT_COST_MODEL_E_V3)
}

#[expect(clippy::unwrap_used)]
fn evaluate(version: &str, body: &str, costs: fn() -> CostModel) -> Outcome {
    let arena = Arena::new();
    let source = format!("(program {version} {body})");
    let program: &Program<'_, DeBruijn> = parse_program(&arena, &source, PROTOCOL_VERSION_12).into_result().unwrap();
    outcome(&program.eval(&arena, costs(), ExBudget::max()))
}

// The second branch rebuilds the scrutinee from its tag and the list it receives.
const SELECTS_FIELDS: &str = "(case (con data (Constr 1 [I 5, B #01])) (error) (lam fields \
    (case [(builtin equalsData) [(builtin constrData) (con integer 1) fields] (con data (Constr 1 [I 5, B #01]))] \
    (error) (con unit ()))))";

#[test]
fn case_on_constr_data_applies_the_tag_branch_to_its_fields() {
    for costs in [v4, v3_at_protocol_12] {
        assert_eq!(evaluate("1.2.0", SELECTS_FIELDS, costs), Outcome::Unit);
    }
}

#[test]
fn case_on_data_is_refused_before_program_version_1_2_0() {
    assert_eq!(evaluate("1.1.0", SELECTS_FIELDS, v4), Outcome::NonConstrScrutinized);
}

#[test]
fn case_on_data_is_refused_before_protocol_version_12() {
    assert_eq!(evaluate("1.2.0", SELECTS_FIELDS, v3_at_protocol_11), Outcome::NonConstrScrutinized);
}

#[test]
fn case_on_constr_data_without_a_branch_for_its_tag_is_refused() {
    assert_eq!(
        evaluate("1.2.0", "(case (con data (Constr 1 [])) (lam fields (con unit ())))", v4),
        Outcome::MissingCaseBranch
    );
}

#[test]
fn case_on_map_data_is_refused() {
    assert_eq!(evaluate("1.2.0", "(case (con data (Map [])) (con unit ()))", v4), Outcome::NonConstrScrutinized);
}

#[test]
fn case_on_list_data_is_refused() {
    assert_eq!(evaluate("1.2.0", "(case (con data (List [])) (con unit ()))", v4), Outcome::NonConstrScrutinized);
}

#[test]
fn case_on_integer_data_is_refused() {
    assert_eq!(evaluate("1.2.0", "(case (con data (I 0)) (con unit ()))", v4), Outcome::NonConstrScrutinized);
}

#[test]
fn case_on_bytes_data_is_refused() {
    assert_eq!(evaluate("1.2.0", "(case (con data (B #)) (con unit ()))", v4), Outcome::NonConstrScrutinized);
}

#[test]
fn a_v4_guarding_script_runs_in_the_ledger_units() {
    let case: serde_json::Value = serde_json::from_str(include_str!("v4_guarding_case.json")).unwrap();
    let integers =
        |key: &str| -> Vec<i64> { case[key].as_array().unwrap().iter().map(|cost| cost.as_i64().unwrap()).collect() };
    let arena = Arena::new();
    let script = hex::decode(case["script"].as_str().unwrap()).unwrap();
    let flat_bytes: minicbor::bytes::ByteVec = minicbor::decode(&script).unwrap();
    let (program, _): (&Program<'_, DeBruijn>, _) =
        flat::decode(&arena, &flat_bytes, PlutusVersion::V3, PROTOCOL_VERSION_12).unwrap();
    let program = case["args"].as_array().unwrap().iter().fold(program, |program, arg| {
        let bytes = hex::decode(arg.as_str().unwrap()).unwrap();
        program.apply(&arena, Term::data(&arena, PlutusData::from_cbor(&arena, &bytes).unwrap()))
    });
    let max_units = integers("max_units");
    let budget = ExBudget { mem: max_units[0], cpu: max_units[1] };
    let result = program.eval(&arena, CostModel::v4(&integers("cost_model")), budget);
    let ran = case["haskell"]["ran"].as_array().unwrap();
    let ledger = ExBudget { mem: ran[0].as_i64().unwrap(), cpu: ran[1].as_i64().unwrap() };
    assert_eq!((outcome(&result), result.info.consumed_budget), (Outcome::Unit, ledger));
}
