"""ECS_C external LoCoMo intake: independent labels, NO science execution.

Pins one historical upstream Git blob; creates prediction-facing DEV-only questions
and transcript snapshots without ever exposing gold answer/evidence fields there.
Raw gold remains separately sealed by convention, not by OS confidentiality.
The pin certifies bytes, not annotation accuracy or independent adjudication.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

SOURCE_COMMIT = '3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376'
GIT_BLOB_SHA1 = 'd95b872480b413d935821fdc3c84f8a8f5f29e73'
SOURCE_URL = f'https://raw.githubusercontent.com/snap-research/locomo/{SOURCE_COMMIT}/data/locomo10.json'
SCHEMA = 'elpis.ecsc.external-locomo-intake.v1'
MAX_BYTES = 8 * 1024 * 1024
DEV_IDS = ('conv-26', 'conv-30')
EXPECTED_SAMPLE_COUNT = 10
EXPECTED_QA_COUNT = 1986
EXPECTED_ABSTENTION_MISSING_ANSWER = 444
EXPECTED_UNRESOLVED_EVIDENCE_REFERENCES = 9


class IntakeError(Exception):
    pass


def demand(condition, code):
    if not condition:
        raise IntakeError(code)


def wire(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=True, indent=2) + '\n').encode('ascii')


def gitblob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _bounded_json(blob):
    demand(type(blob) is bytes and len(blob) <= MAX_BYTES, 'SOURCE_SIZE_INVALID')
    try:
        data = json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntakeError('SOURCE_JSON_INVALID') from exc
    demand(type(data) is list, 'SOURCE_SHAPE_INVALID')
    return data


def convert(blob, *, enforce_release=True):
    """Keep every question; expose only DEV transcripts/questions to candidate systems.

    QUAL records are withheld entirely from candidate input. Public source
    is still available elsewhere; no claims of hard access isolation.
    """
    if enforce_release:
        demand(gitblob(blob) == GIT_BLOB_SHA1, 'SOURCE_PIN_MISMATCH')
    raw = _bounded_json(blob)
    all_ids = set()
    public_dev = []
    gold = []
    anomalies = []
    count = missing_answer = evidence_issues = 0
    for sample in raw:
        demand(type(sample) is dict and type(sample.get('sample_id')) is str, 'SAMPLE_INVALID')
        sid = sample['sample_id']
        demand(sid not in all_ids and re.fullmatch(r'conv-\d+', sid) is not None, 'SAMPLE_ID_INVALID')
        all_ids.add(sid)
        convo, questions = sample.get('conversation'), sample.get('qa')
        demand(type(convo) is dict and type(questions) is list, 'SAMPLE_PAYLOAD_INVALID')
        speaker_a, speaker_b = convo.get('speaker_a'), convo.get('speaker_b')
        demand(all(type(x) is str and x for x in (speaker_a, speaker_b)), 'SPEAKER_METADATA_INVALID')
        names = sorted((k for k in convo if re.fullmatch(r'session_\d+', k)), key=lambda x:int(x.split('_')[1]))
        demand(names and len(names) == len({int(k.split('_')[1]) for k in names}), 'SESSIONS_INVALID')
        turns = []
        dialog_ids = set()
        for name in names:
            stamp = convo.get(name + '_date_time')
            demand(type(stamp) is str and stamp, 'TIMESTAMP_INVALID')
            session = convo[name]
            demand(type(session) is list and session, 'SESSION_EMPTY')
            public_turns = []
            for item in session:
                demand(type(item) is dict and all(type(item.get(k)) is str for k in ('dia_id','speaker','text')),
                       'TURN_INVALID')
                did = item['dia_id']
                demand(did and did not in dialog_ids, 'DUPLICATE_DIALOG_ID')
                dialog_ids.add(did)
                # Ignore visual metadata: images not released with original benchmark.
                public_turns.append({'dialog_id': did, 'speaker': item['speaker'], 'text': item['text']})
            turns.append({'session': name, 'date_time': stamp, 'turns': public_turns})
        public_questions = []
        for qi, question in enumerate(questions):
            demand(type(question) is dict and type(question.get('question')) is str and question['question'],
                   'QUESTION_INVALID')
            cat = question.get('category')
            demand(type(cat) is int and cat in (1,2,3,4,5), 'CATEGORY_INVALID')
            qid = sid + '/q' + str(qi).zfill(4)
            public_questions.append({'id': qid, 'question': question['question'], 'category': cat})
            evidence = question.get('evidence', [])
            demand(type(evidence) is list and all(type(e) is str for e in evidence), 'EVIDENCE_FORMAT_INVALID')
            unmatched = [e for e in evidence if e not in dialog_ids]
            for e in unmatched:
                anomalies.append({'id': qid, 'type':'UNRESOLVED_EVIDENCE_ID', 'raw':e})
            evidence_issues += len(unmatched)
            if 'answer' not in question:
                missing_answer += 1
                if cat != 5:
                    anomalies.append({'id':qid, 'type':'MISSING_NON_ABSTENTION_ANSWER'})
            else:
                demand(type(question['answer']) in (str,int,float) and type(question['answer']) is not bool,
                       'ANSWER_INVALID')
            gold.append({'id':qid, 'sample_id':sid, 'category':cat,
                         'answer_present': 'answer' in question,
                         'answer': question.get('answer'),
                         'adversarial_answer':question.get('adversarial_answer'),
                         'evidence_original':evidence,
                         'evidence_unresolved':unmatched})
            count += 1
        if sid in DEV_IDS:
            public_dev.append({'sample_id':sid, 'speakers':[speaker_a,speaker_b],
                               'sessions':turns, 'questions':public_questions})
    if enforce_release:
        demand(len(raw)==EXPECTED_SAMPLE_COUNT and count==EXPECTED_QA_COUNT, 'RELEASE_COUNTS_UNEXPECTED')
        demand(missing_answer == EXPECTED_ABSTENTION_MISSING_ANSWER, 'ABSTENTION_COUNT_DRIFT')
        demand(evidence_issues==EXPECTED_UNRESOLVED_EVIDENCE_REFERENCES, 'EVIDENCE_ISSUE_COUNT_DRIFT')
        demand(set(DEV_IDS).issubset(all_ids), 'DEV_IDS_MISSING')
    demand(len(public_dev)==len(DEV_IDS) if enforce_release else True, 'DEV_SPLIT_INVALID')
    # Independent-source labels exist, but these are authored benchmark annotations,
    # not verified semantic entailments or any demonstrated Elpis capability.
    summary={'schema':SCHEMA, 'status':'INTAKE_ONLY_NOT_SCIENTIFIC_QUAL',
             'source_commit': SOURCE_COMMIT if enforce_release else 'UNIT_TEST_ONLY',
             'source_git_blob_sha1':gitblob(blob), 'source_sha256':digest(blob),
             'sample_count':len(raw), 'question_count':count,
             'missing_answer_count':missing_answer,
             'unresolved_evidence_reference_count':evidence_issues,
             'anomalies':anomalies,
             'dev_sample_ids':sorted(set(all_ids).intersection(DEV_IDS)),
             'qual_sample_ids':sorted(all_ids.difference(DEV_IDS)),
             'qual_status':'SEALED_BY_PROTOCOL_ONLY_NOT_ACL_ENFORCED',
             'semantics':'NONE_PERFORMED',
             'prohibition':'NO_QUAL_INPUT_OR_GOLD_TO_PREDICTORS_NO_POSTHOC_REPAIR'}
    # Return only prediction-facing dev data plus all label data kept in a separate file;
    # the caller must never pass the gold to predictors.
    pub={'schema':SCHEMA+'.public-dev', 'status':'DEV_ONLY', 'samples':public_dev}
    private={'schema':SCHEMA+'.gold', 'status':'SCORER_ONLY', 'annotations':gold}
    return pub, private, summary


def safe_write(path, content, *, private=False):
    """No replacement or repair of historical evidence. A failed process leaves no partial output."""
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        demand(path.is_file() and not path.is_symlink() and path.read_bytes()==content,
               'EXISTING_OUTPUT_CONFLICT:'+path.name)
        return False
    temp=path.with_name('.'+path.name+'.tmp.'+str(os.getpid()))
    try:
        fd=os.open(temp, os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW, 0o600 if private else 0o644)
        try:
            with os.fdopen(fd,'wb') as f:
                f.write(content)
                f.flush();os.fsync(f.fileno())
        except BaseException:
            raise
        # Hard-link installs only if absent; no TOCTOU overwrite.
        os.link(temp,path)
    except FileExistsError as exc:
        demand(path.is_file() and not path.is_symlink() and path.read_bytes()==content,
               'EXISTING_OUTPUT_CONFLICT:'+path.name)
        return False
    finally:
        temp.unlink(missing_ok=True)
    return True


def get_pinned_source(path):
    path=Path(path)
    if path.exists():
        demand(path.is_file() and not path.is_symlink(), 'SOURCE_PATH_INVALID')
        raw=path.read_bytes()
        demand(gitblob(raw)==GIT_BLOB_SHA1, 'SOURCE_PIN_MISMATCH')
        print('SOURCE=PINNED_LOCAL', flush=True)
        return raw
    print('SOURCE=FETCH_UPSTREAM_COMMIT_PINNED (2.8 MB maximum 8 MB)',flush=True)
    # No dependency on /home/joe; write only in permanent research directory.
    req=urllib.request.Request(SOURCE_URL,headers={'User-Agent':'ECSC-Dev-Intake/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            demand(resp.geturl()==SOURCE_URL, 'SOURCE_REDIRECT_REFUSED')
            length=resp.headers.get('Content-Length')
            if length is not None:
                demand(int(length) <= MAX_BYTES, 'SOURCE_OVERSIZE')
            buf=bytearray()
            while True:
                part=resp.read(256*1024)
                if not part:break
                buf.extend(part)
                demand(len(buf)<=MAX_BYTES, 'SOURCE_OVERSIZE')
                print('SOURCE_FETCH_BYTES=',len(buf),flush=True)
    except Exception as exc:
        raise IntakeError('SOURCE_FETCH_FAILED:'+type(exc).__name__+':'+str(exc)[:140]) from exc
    raw=bytes(buf)
    demand(gitblob(raw)==GIT_BLOB_SHA1, 'SOURCE_PIN_MISMATCH')
    safe_write(path,raw,private=True)
    return raw


def run(worktree):
    w=Path(worktree)
    research=w/'research/ecsc_hca_r0'
    freeze=research/'R8_DESCRIPTIVE_ACCOUNTING.json'
    demand(freeze.is_file() and digest(freeze.read_bytes()) ==
           'fb71c347b0336eb4aea59cbd5e65b0df3b7d0cb1ac2797d9ad888116d8bc8af5',
           'R8_ACCOUNTING_PIN_MISMATCH')
    target=research/'external/locomo_r0'
    raw=get_pinned_source(target/'locomo10.json')
    public,private,summary=convert(raw)
    for name,obj,hidden in (('DEV_INPUT.json',public,False),('SCORER_GOLD.json',private,True),
                            ('INTAKE_AUDIT.json',summary,False)):
        safe_write(target/name,wire(obj),private=hidden)
    print('R9_SOURCE_SHA256=',summary['source_sha256'],flush=True)
    print('R9_SAMPLES=',summary['sample_count'],'R9_QUESTIONS=',summary['question_count'],flush=True)
    print('R9_MISSING_ANSWERS=',summary['missing_answer_count'],
          'R9_UNRESOLVED_EVIDENCE_REFERENCES=',summary['unresolved_evidence_reference_count'],flush=True)
    print('R9_DEV_CASES=',sum(len(s['questions']) for s in public['samples']),flush=True)
    print('R9_QUAL_SAMPLES=',len(summary['qual_sample_ids']),flush=True)
    print('R9_EXTERNAL_INTAKE=PASS_INTAKE_ONLY_NOT_SCIENTIFIC_QUAL',flush=True)
    return summary


if __name__ == '__main__':
    try:
        run(Path(__file__).resolve().parents[2])
    except Exception as exc:
        print('R9_EXTERNAL_INTAKE=HOLD',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        raise SystemExit(3)
