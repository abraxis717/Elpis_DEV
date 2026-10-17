"""R13 DEV diagnostic: source-content perturbation versus B0 typed structural readout.

A source sentence's negation is changed, while source verification and relation-order
proofs are regenerated independently for each temporary copy. The current B0
pipeline sees source identities and supplied typed relations, not semantic tokens.
NO semantic success, learning, independent QA, or QUAL access is claimed.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

from .baselines import Baselines, Refusal
from .r4_hacf_sources import verified_source_checks
from .r9_external_intake import digest, safe_write, wire
from .r11_candidate_graph import derived_manifest, load_predictor_inputs, source_ranges
from .r12_native_structure import (
    LIMITS, NativeRefusal, directional_predictions, selected_centers, source_graph,
)

SCHEMA = 'elpis.ecsc.r13.native-semantic-ingress-null-diagnostic.v1'
DEV_PIN = '89b8ed64d9a3cf83b7efb87611cef017dc65a2775a634851ecd6e6f711f50672'
PROBE_TEXT = {
    'AFFIRMATIVE': 'The amber key belongs to Mira.',
    'NEGATED': 'The amber key does not belong to Mira.',
}


def demand(ok, code):
    if not ok:
        raise NativeRefusal('R13_' + code)


def alter_one_turn(dev, sample_id, turn_id, text):
    """Produce a distinct, in-memory DEV copy; never change the frozen source."""
    demand(type(text) is str and len(text) <= 256, 'TEXT_SHAPE')
    altered = deepcopy(dev)
    replaced = 0
    for sample in altered['samples']:
        if sample['sample_id'] != sample_id:
            continue
        for session in sample['sessions']:
            for turn in session['turns']:
                if turn['dialog_id'] == turn_id:
                    turn['text'] = text
                    replaced += 1
    demand(replaced == 1, 'MISSING_OR_DUPLICATE_CENTER')
    return altered


def inspected_sample(dev, sid):
    matches = [sample for sample in dev['samples'] if sample['sample_id'] == sid]
    demand(len(matches) == 1, 'SAMPLE_LOOKUP')
    return matches[0]


def _build(source_root, dev, sid, triple):
    raw = wire(dev)
    f = source_root / 'DEV_INPUT.json'
    demand(not f.exists(), 'TEMPORARY_FILE_ALREADY_EXISTS')
    f.write_bytes(raw)
    spans, n = source_ranges(raw, dev)
    manifest = derived_manifest(raw, n)
    sample = inspected_sample(dev, sid)
    view, auth, bindings, axioms, scope = source_graph(sample, spans, raw, manifest, triple)
    checks = verified_source_checks(source_root, manifest, view, auth, bindings,
        limits=LIMITS, max_document_bytes=1<<20, max_total_document_bytes=1<<20)
    demand(len(checks) == 3, 'SOURCE_COVERAGE')
    graph = Baselines.build(view, auth, checks, LIMITS)
    output, work = directional_predictions(graph, scope, axioms, triple[1])
    demand(output == {'PREVIOUS':triple[0], 'NEXT':triple[2]}, 'STRUCTURAL_READOUT')
    return {
        'output': output,
        'document_digest': digest(raw),
        'center_chunk_digest': next(b.claim.chunk_digest for b in bindings
                                    if b.reference == axioms[triple[1]].references[0]),
        'source_verified': len(checks),
        'logical_work': sum(work.values()),
        'readout_claim': 'SALIENCE_ONLY',
    }, (manifest, view, auth, bindings)


def compare_one(dev, sid, triple, temp_parent):
    variants = {}
    for label, text in PROBE_TEXT.items():
        changed = alter_one_turn(dev, sid, triple[1], text)
        with TemporaryDirectory(prefix='.r13_probe_', dir=temp_parent) as dirname:
            source_root = Path(dirname)
            result, refs = _build(source_root, changed, sid, triple)
            # Mutate ONLY the temporary source file after its verified graph is prepared.
            # Reusing the previous manifest and claims must fail closed, not use stale bytes.
            other = PROBE_TEXT['NEGATED' if label == 'AFFIRMATIVE' else 'AFFIRMATIVE']
            altered_again = alter_one_turn(dev, sid, triple[1], other)
            (source_root/'DEV_INPUT.json').write_bytes(wire(altered_again))
            manifest, view, auth, bindings = refs
            refused = False
            try:
                verified_source_checks(source_root, manifest, view, auth, bindings,
                    limits=LIMITS, max_document_bytes=1<<20,
                    max_total_document_bytes=1<<20)
            except Refusal:
                refused = True
            demand(refused, 'STALE_SOURCE_WAS_ACCEPTED')
            result['stale_source_refused'] = refused
            variants[label] = result
    a, b = variants['AFFIRMATIVE'], variants['NEGATED']
    demand(a['document_digest'] != b['document_digest'], 'DOCUMENT_IDENTITY_COLLAPSE')
    demand(a['center_chunk_digest'] != b['center_chunk_digest'], 'CHUNK_IDENTITY_COLLAPSE')
    demand(a['output'] == b['output'], 'UNEXPECTED_STRUCTURAL_CHANGE')
    return {'sample_id': sid, 'center': triple[1],
            'structural_destinations': a['output'],
            'source_content_changed': True,
            'source_checks_recomputed': True,
            'same_structural_output': True,
            'stale_source_refused': True,
            'affirmative_source_sha256': a['document_digest'],
            'negated_source_sha256': b['document_digest'],
            'logical_work_by_variant': {label:variants[label]['logical_work'] for label in PROBE_TEXT}}


def run(work):
    source_root, dev, _, dev_sha = load_predictor_inputs(work)
    demand(dev_sha == DEV_PIN, 'DEV_PIN_MISMATCH')
    report_path = source_root / 'R13_DEV_INGRESS_NULL_REPORT.json'
    original_file = source_root / 'DEV_INPUT.json'
    demand(original_file.is_file() and not original_file.is_symlink(), 'DEV_SOURCE_MISSING')
    original_hash = digest(original_file.read_bytes())
    demand(original_hash == DEV_PIN, 'DEV_SOURCE_DRIFT')
    cases = []
    for sample in dev['samples']:
        triples = selected_centers(sample, 1)
        case = compare_one(dev, sample['sample_id'], triples[0], source_root)
        cases.append(case)
        print('R13_CONTENT_PAIR=', len(cases), 'CENTER=', case['center'],
              'SAME_STRUCTURAL_OUTPUT=', case['same_structural_output'],
              'STALE_SOURCE_REFUSED=', case['stale_source_refused'], flush=True)
    demand(len(cases) == 2, 'CASE_COUNT')
    demand(digest(original_file.read_bytes()) == original_hash, 'FROZEN_DEV_MODIFIED')
    report = {
        'schema': SCHEMA,
        'status': 'PASS_NATIVE_CONTENT_SENSITIVITY_NULL_DIAGNOSTIC_DEV_ONLY',
        'frozen_dev_sha256': original_hash,
        'cases': cases,
        'pairs': 2,
        'real_model': 'ELPIS_B0_NO_EXTERNAL_LLM',
        'conclusion': 'With typed relation topology held constant, changing verified source sentence content does not change R12N structural B0 destinations.',
        'interpretation': 'Expected by design: B0 has no text-semantic projection. This is NOT a test of factual question-answering accuracy.',
        'semantic_qualification': False,
        'qual_opened': False,
    }
    encoded = wire(report)
    safe_write(report_path, encoded)
    print('R13_REPORT=', report_path, 'SHA256=', digest(encoded), flush=True)
    print('R13_INGRESS_NULL=PASS_DIAGNOSTIC_ONLY_NOT_SEMANTIC_QUAL', flush=True)
    return report


if __name__ == '__main__':
    try:
        demand(len(sys.argv) == 2, 'USAGE_WORKTREE')
        run(Path(sys.argv[1]))
    except Exception as exc:
        print('R13_INGRESS_NULL=HOLD', type(exc).__name__, str(exc), file=sys.stderr, flush=True)
        sys.exit(3)
