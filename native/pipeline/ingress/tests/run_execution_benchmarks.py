"""Reproducible process-isolated direct/1/2/3/4-worker Regex trials.

python native/pipeline/ingress/tests/run_execution_benchmarks.py \
  --binary BUILD/native/pipeline/benchmark_regex_execution --output RESULTS

Run after compiler/test jobs finish. No third-party Python packages required.
Every raw JSON result includes the exact command; compare medians, not best runs.
"""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
if args.repeats < 1:
    parser.error('positive repeats required')
args.output.mkdir(parents=True, exist_ok=True)
binary = args.binary.resolve()
rows = []
for family, count in [('small', 300), ('large', 12)]:
    expected = None
    for repeat in range(args.repeats):
        # Rotate order to avoid always assigning the coldest/earliest run to 0.
        modes = list(range(5))
        modes = modes[repeat % 5:] + modes[:repeat % 5]
        for workers in modes:
            command = [str(binary), str(workers), family, str(count)]
            run = subprocess.run(command, text=True, capture_output=True, timeout=600)
            name = f'{family}-r{repeat}-w{workers}'
            (args.output / (name+'.stdout')).write_text(run.stdout)
            (args.output / (name+'.stderr')).write_text(run.stderr)
            if run.returncode:
                raise SystemExit(f'{command}: exit {run.returncode}: {run.stderr}')
            row = json.loads(run.stdout)
            row.update(command=command, repeat=repeat)
            invariant = (row['source_bytes'], row['output_bytes'], row['failures'])
            if expected is None:
                expected = invariant
            assert invariant == expected, (expected, invariant, command)
            rows.append(row)
            (args.output / 'raw.json').write_text(json.dumps(rows, indent=2)+'\n')
            print(json.dumps(row), flush=True)
