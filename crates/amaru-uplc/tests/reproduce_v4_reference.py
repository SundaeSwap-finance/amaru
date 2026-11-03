#!/usr/bin/env python3
"""Offline V4 reference checks; requires the Pallas registration export (see v4-reference.md)."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cli', required=True, type=Path)
parser.add_argument('--uplc', required=True, type=Path)
parser.add_argument('--pallas', required=True, type=Path)
parser.add_argument('--export-dir', required=True, type=Path)
args = parser.parse_args()
fixtures = Path(__file__).resolve().parent
provenance = json.loads((fixtures / 'v4_provenance.json').read_text())
for binary, key in [(args.cli, 'cli_sha256'), (args.uplc, 'uplc_binary_sha256')]:
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == provenance['reference'][key], binary
spec = importlib.util.spec_from_file_location('raw_cbor', args.pallas / 'test_data/musashi-guard-reference/reproduce.py')
cbor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cbor)
out = args.export_dir
save = lambda name, value: (out / name).write_text(json.dumps(value, indent=2) + '\n')
original = json.loads((out / 'settings.tx.json').read_text())
raw = bytes.fromhex(original['cborHex'])
tx, end = cbor.decode(raw)
assert end == len(raw) and cbor.encode(tx) == raw
assert tx.value[0].raw == bytes.fromhex((args.pallas / 'test_data/musashi-registration/settings.body.hex').read_text())
body = tx.value[0].value
assert 3 not in body and 8 not in body
parameters = cbor.parameters(args.pallas)
genesis = json.loads((args.pallas / 'test_data/musashi-ttl/genesis/dijkstra.json').read_text())
for key, value in genesis.items():
    if key not in parameters and key != 'plutusV4CostModel':
        parameters[key] = value
assert parameters['costModels']['PlutusV4'] == json.loads((fixtures / 'v4_costs.json').read_text())
save('v4-parameters.json', parameters)
history = [[[0, 0, 0], [0, 0, 0], [21600, 10**12, [0, 432, [0]], 432]] for _ in range(7)]
history.append([[0, 0, 0], None, [21600, 10**12, [1], 432]])
save('v4-era-history.json', {'type': 'EraHistory', 'description': 'Synthetic; unbounded probe transactions', 'cborHex': cbor.encode(history).hex()})
results = {}
for name, case in provenance['scripts'].items():
    script = bytes.fromhex(case['cbor_hex'])
    assert hashlib.sha256(script).hexdigest() == case['decoded_cbor_sha256']
    script_hash = hashlib.blake2b(b'\x04' + script, digest_size=28).digest()
    assert script_hash.hex() == case['script_hash']
    changed = body.copy()
    changed[4] = cbor.Tag(258, [[7, [1, script_hash], 2000000]])
    witnesses = tx.value[1].value.copy()
    witnesses[8] = cbor.Tag(258, [script])
    save('v4-' + name + '.tx.json', {'type': 'Tx DijkstraEra', 'description': 'Synthetic phase-two probe; invalid original signatures; never submitted', 'cborHex': cbor.encode([changed, witnesses, *tx.value[2:]]).hex()})
    command = [str(args.cli), 'dijkstra', 'transaction', 'calculate-plutus-script-cost', 'offline',
               '--start-time-utc', '2026-09-07T00:00:00Z', '--era-history-file', str(out / 'v4-era-history.json'),
               '--protocol-params-file', str(out / 'v4-parameters.json'), '--utxo-file', str(out / 'settings.utxos.json'),
               '--tx-file', str(out / ('v4-' + name + '.tx.json'))]
    r = subprocess.run(command, capture_output=True, text=True)
    results[name] = {'exit_code': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
    assert (r.returncode == 0) == case['success'], results[name]
    if case['success']:
        assert json.loads(r.stdout)[0]['executionUnits'] == case['execution_units']
    else:
        assert case['error_contains'] in r.stderr
    print(name + ': expected V4 reference result')
# Explicitly synthetic model perturbations: V4 must use its own supplied costs.
for language, cpu in [('PlutusV3', 64100), ('PlutusV4', 65100)]:
    changed_parameters = json.loads(json.dumps(parameters))
    changed_parameters['costModels'][language][29] += 1000
    parameter_path = out / ('v4-perturbed-' + language + '.json')
    save(parameter_path.name, changed_parameters)
    check = command.copy()
    check[check.index('--protocol-params-file') + 1] = str(parameter_path)
    check[check.index('--tx-file') + 1] = str(out / 'v4-unit.tx.json')
    r = subprocess.run(check, capture_output=True, text=True)
    results['model-' + language] = {'exit_code': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
    assert r.returncode == 0 and json.loads(r.stdout)[0]['executionUnits'] == {'memory': 500, 'steps': cpu}
for name in ['unit', 'grow']:
    source = '(program 1.1.0 [(lam ctx (con unit ())) (con data (I 0))])\n' if name == 'unit' else (fixtures / 'v4_grow.uplc').read_text()
    path = out / ('v4-' + name + '-applied.uplc')
    path.write_text(source)
    units = provenance['scripts'][name]['execution_units']
    for label, dcpu, dmem in [('exact', 0, 0), ('cpu-short', 1, 0), ('memory-short', 0, 1)]:
        budget = f"{units['steps'] - dcpu}:{units['memory'] - dmem}"
        r = subprocess.run([str(args.uplc), 'evaluate', '-S', 'G', '-R', budget, '-i', str(path)], capture_output=True, text=True)
        results[name + '-' + label] = {'exit_code': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
        assert (r.returncode == 0) == (label == 'exact')
save('v4-reference-results.json', results)
print('All six exact/short-budget controls matched')
