# Protocol-12 V4/G backend reference

`CostModel::v4` opts into G. Existing V1–V3 selectors keep their historical
behavior. G retains E's costs, UTF-8 costing, bytestring operand limits and
strict `consByteString` conversion, while removing the `CInteger`
operand bound for arithmetic and ordering. The pinned reference sources and tool hashes are in
[v4_provenance.json](v4_provenance.json).

The new CLI independently evaluates native Dijkstra **V4** witnesses. Its unit
script costs 500 memory / 64,100 CPU; its compact 130-byte Flat script repeatedly
squares 2, adds zero to the large result, and returns unit for 26,424 memory /
2,912,407,587 CPU. Both match the backend exactly. The same arithmetic assertions
fail with the previous E selector because it rejects the large operand.

The 251 cost coefficients are extracted unchanged from Musashi's historical
parameters. They match the reference defaults' prefix. Reference V4 has 369
names: the first 350 match the existing V3 map; the remaining 19 price future
builtins unavailable at protocol 12. Missing names receive maximal costs.
No model is shortened or replaced to obtain a passing test. Separate synthetic
controls increase startup CPU by 1,000: changing the V4 model raises the V4 unit
cost to 65,100; changing only the V3 model leaves it at 64,100.

The captured short vector leaves `insertCoin` unpriced. The CLI rejects that
case for budget exhaustion. Four-argument linear costing now saturates like the
other cost functions, preventing an overflow panic (or wrapping release cost).

The Rust tests cover exact and insufficient budgets, explicit failure, all seven
arithmetic operations and both ordering comparisons with oversized operands in either position/sign, retained
bytestring/byte-value limits, UTF-8 byte costing, missing costs and future parameter tails. Existing
conformance tests protect the previous selectors.

The reference checks cover the stated operations; they do not establish complete
V4 conformance or newly audit every builtin.

This is **raw evaluator/cost-model support**, not ledger validation. Raw UPLC can
return non-unit values; the ledger API separately requires unit. Language
availability, script decoding, V4 context construction and result validation
remain caller responsibilities. Guarding dispatch, datums, batches and child
execution are outside this change. Pallas/Dolos production pins are not changed
by these tests.

## Reproduction

Run `cargo test -p amaru-uplc --test v4_semantics` and `cargo test -p amaru-uplc`.
The reference script uses the public Pallas capture without copying entire
transactions into this crate. With Pallas at the revision in provenance, first
export its original registration (from the Pallas checkout):

```sh
MUSASHI_EVAL_EXPORT=/tmp/v4-reference-export \
  cargo test --offline -p pallas-validate --features phase2,unstable \
  export_registration_cli_comparison -- --ignored
```

Then, from this checkout, run with the two pinned reference executables:

```sh
python3 crates/amaru-uplc/tests/reproduce_v4_reference.py \
  --pallas /path/to/pallas --cli /path/to/cardano-cli \
  --uplc /path/to/uplc-1.71 --export-dir /tmp/v4-reference-export
```

The original registration is for RealFi's settings validator, an application
configuration script. Probe scripts/certificates/witnesses are synthetic and make
its original signatures invalid. Original fixture bytes are never edited. Missing
Peras/reference-input parameter fields use retained configured genesis values;
historical values are unknown and irrelevant to these context-ignoring programs.
The synthetic era interpreter cannot affect transactions with no validity bounds.
Full CLI outputs and retry history belong outside Git. No transaction was funded,
signed or submitted, and no deployed-node build or on-chain acceptance is claimed.
