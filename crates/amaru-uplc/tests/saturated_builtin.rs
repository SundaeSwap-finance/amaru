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

use amaru_kernel::protocol_version::PROTOCOL_VERSION_11;
use amaru_uplc::{
    arena::Arena,
    binder::DeBruijn,
    builtin::DefaultFunction,
    constant::Constant,
    machine::{CostModel, ExBudget, Machine, env::Env, state::MachineState, value::Value},
    program::Program,
    syn::parse_program,
};

struct Run {
    saturated: Vec<DefaultFunction>,
    logs: usize,
    integer: Option<i64>,
}

#[expect(clippy::unwrap_used)]
fn run(body: &str) -> Run {
    let arena = Arena::new();
    let source = format!("(program 1.1.0 {body})");
    let program: &Program<'_, DeBruijn> = parse_program(&arena, &source, PROTOCOL_VERSION_11).into_result().unwrap();
    let mut machine = Machine::new(&arena, ExBudget::max(), CostModel::v3());
    let mut state = MachineState::compute(0, Env::new_in(&arena), program.term);
    let mut saturated = Vec::new();
    let value = loop {
        state = match state {
            MachineState::Compute(context, env, term) => machine.compute(context, env, term).unwrap(),
            MachineState::Return(context, value) => {
                saturated.extend(machine.saturated_builtin(value));
                machine.return_compute(context, value).unwrap()
            }
            MachineState::Done(value) => break value,
        };
    };
    let integer = if let Value::Con(Constant::Integer(integer)) = value { i64::try_from(*integer).ok() } else { None };
    Run { saturated, logs: machine.info().logs.len(), integer }
}

#[test]
fn a_partial_application_saturates_nothing() {
    for body in ["(builtin addInteger)", "[(builtin addInteger) (con integer 1)]", "(force (builtin headList))"] {
        assert_eq!(run(body).saturated, [], "{body}");
    }
}

#[test]
fn applying_the_last_argument_saturates_the_builtin() {
    let run = run("[(builtin addInteger) (con integer 1) (con integer 2)]");
    assert_eq!((run.saturated, run.integer), (vec![DefaultFunction::AddInteger], Some(3)));
}

#[test]
fn a_forced_builtin_saturates_on_its_argument() {
    let run = run("[(force (builtin headList)) (con (list integer) [7])]");
    assert_eq!((run.saturated, run.integer), (vec![DefaultFunction::HeadList], Some(7)));
}

#[test]
fn a_case_branch_saturates_on_the_constr_fields() {
    let run = run("(case (constr 0 (con integer 1) (con integer 2)) (builtin addInteger))");
    assert_eq!((run.saturated, run.integer), (vec![DefaultFunction::AddInteger], Some(3)));
}

#[test]
fn each_trace_call_is_saturated_once() {
    let run =
        run("[(force (builtin trace)) (con string \"a\") [(force (builtin trace)) (con string \"b\") (con unit ())]]");
    assert_eq!((run.saturated, run.logs), (vec![DefaultFunction::Trace, DefaultFunction::Trace], 2));
}
