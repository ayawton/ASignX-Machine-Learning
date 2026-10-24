"""Merge completed parallel replay caches and publish source evidence safely."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/source-recovery'
CACHE = OUT / 'exhaustive'


def main():
    states = [json.loads((CACHE / f'archive3-{folder}-progress.json').read_text(encoding='utf-8')) for folder in 'ABCDE']
    assert all(s['complete'] for s in states)
    assert len({s['dataset_sha256'] for s in states}) == 1
    base = CACHE / 'archive3.sqlite3'
    backup = CACHE / 'archive3.before-partition-merge.sqlite3'
    assert not backup.exists()
    shutil.copyfile(base, backup)
    with sqlite3.connect(base) as dest:
        for folder in 'ABCDE':
            with sqlite3.connect(CACHE / f'archive3-{folder}.sqlite3') as src:
                partition = json.loads(src.execute("SELECT value FROM meta WHERE key='inputs'").fetchone()[0])
                del partition['subject_folder_partition']
                assert partition == json.loads(dest.execute("SELECT value FROM meta WHERE key='inputs'").fetchone()[0])
                dest.executemany('INSERT OR IGNORE INTO images VALUES(?,?,?,?)', src.execute('SELECT * FROM images'))
                dest.executemany('INSERT OR IGNORE INTO matches VALUES(?,?,?,?,?)', src.execute('SELECT * FROM matches'))
        dest.commit()
        count = dest.execute('SELECT count(*) FROM images').fetchone()[0]
        assert count == sum(s['total_images'] for s in states) == 65774
        state = dict(states[0], total_images=count, processed_images=count, complete=True,
                     unique_current_rows_matched=dest.execute('SELECT count(DISTINCT csv_row) FROM matches').fetchone()[0],
                     partitioned_execution=True, partitions=states)
        state.pop('elapsed_seconds_this_process', None)
        state.pop('images_per_second', None)
        (CACHE / 'archive3-progress.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    summary = {'dataset_sha256': states[0]['dataset_sha256'], 'archives': {}, 'signer_independence_verified': False,
               'implementation_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   [ROOT / name for name in ('tmp/recover_sources_exhaustive.py', 'asignx/detector.py',
                                            'ml/extract_dataset.py', 'ml/preprocessing.py', 'ml/data_io.py')]},
               'recovery_configuration': {'static_image_mode': True, 'max_num_hands': 2,
                                          'min_detection_confidence': 0.5, 'require_single_hand': True,
                                          'retry_path': 'ml.extract_dataset._detect_hand_with_retries'}}
    scripts = CACHE / 'scripts'
    scripts.mkdir(exist_ok=True)
    for name in ('recover_sources_exhaustive.py', 'finalize_source_recovery.py',
                 'train_source_separated.py', 'complete_source_experiment.py'):
        shutil.copyfile(ROOT / 'tmp' / name, scripts / name)
    matches = defaultdict(lambda: defaultdict(list))
    with (CACHE / 'image-row-matches.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['archive_family', 'archive_member', 'image_sha256', 'current_csv_row', 'max_coordinate_delta', 'handedness', 'source_subject_folder'])
        for family in ('archive3', 'archive5'):
            state = json.loads((CACHE / f'{family}-progress.json').read_text(encoding='utf-8'))
            assert state['complete']
            with sqlite3.connect(CACHE / f'{family}.sqlite3') as db:
                meta = json.loads(db.execute("SELECT value FROM meta WHERE key='inputs'").fetchone()[0])
                assert meta['dataset_sha256'] == summary['dataset_sha256']
                with Path(state['archive']).open('rb') as archive_handle:
                    assert hashlib.file_digest(archive_handle, 'sha256').hexdigest() == meta['archive_sha256']
                summary['archives'][family] = {'images_processed': db.execute('SELECT count(*) FROM images').fetchone()[0],
                    'unique_current_rows_matched': db.execute('SELECT count(DISTINCT csv_row) FROM matches').fetchone()[0],
                    'status_counts': dict(db.execute('SELECT status,count(*) FROM images GROUP BY status')),
                    'inputs': meta}
                for member, sha, row, delta, hand, folder in db.execute('SELECT m.member,i.image_sha256,m.csv_row,m.max_delta,m.handedness,m.subject_folder FROM matches m JOIN images i ON i.member=m.member'):
                    writer.writerow([family, member, sha, row, delta, hand, folder])
                    matches[row][family].append((member, folder))
    assert hashlib.sha256((ROOT / 'ml/data/landmarks.csv').read_bytes()).hexdigest() == summary['dataset_sha256']
    provenance_path = OUT / 'row-provenance.csv'
    shutil.copyfile(provenance_path, OUT / 'row-provenance.before-exhaustive.csv')
    with provenance_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames) + ['image_replay_families', 'image_replay_matches', 'source_subject_folders', 'signer_id', 'identity_status']
        rows = list(reader)
    session_matches = defaultdict(Counter)
    newly_supported = 0
    for row in rows:
        replay = matches[int(row['current_csv_row'])]
        row.update(image_replay_families=';'.join(sorted(replay)), image_replay_matches=sum(map(len, replay.values())),
                   source_subject_folders=';'.join(sorted({folder for members in replay.values() for _, folder in members if folder})),
                   signer_id='', identity_status='unknown')
        for family in replay:
            session_matches[row['session_id']][family] += 1
        if not row['source_id'] and len(replay) == 1:
            family = next(iter(replay))
            row['source_id'] = 'sha256:' + summary['archives'][family]['inputs']['archive_sha256']
            row['evidence'] = 'image_replay_all_63_coordinates_atol_1e-6_same_label_and_hand;not_unique_image_or_signer'
            newly_supported += 1
    with provenance_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    summary['newly_source_supported_rows'] = newly_supported
    summary['source_supported_rows_total'] = sum(bool(r['source_id']) for r in rows)
    summary['current_rows'] = len(rows)
    summary['sessions_with_legacy_image_evidence'] = len(session_matches)

    identity_path = ROOT / 'output/signer-benchmark/identity-plan.csv'
    shutil.copyfile(identity_path, identity_path.with_name('identity-plan.before-exhaustive.csv'))
    with identity_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames; identities = list(reader)
    constraints = 0
    conflicts = []
    for row in identities:
        families = session_matches.get(row['session_id'], {})
        if len(families) == 1 and not row['group_id']:
            family = next(iter(families))
            sha = summary['archives'][family]['inputs']['archive_sha256']
            row.update(group_id='source-family:sha256:' + sha,
                       group_kind='source_co_location_not_verified_person',
                       group_evidence=f'image_replay_rows={families[family]}/{row["samples"]};partial_session_evidence_not_exclusive_attribution')
            archive_name = 'archive (3).zip' if family == 'archive3' else 'archive (5).zip'
            row['source_path'] = str(Path('C:/Users/jibbi/Downloads') / archive_name)
            row['source_evidence'] = row['group_evidence']
            constraints += 1
        elif len(families) > 1:
            conflicts.append({'session_id': row['session_id'], 'matched_families': dict(families)})
    with identity_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(identities)
    summary['new_source_co_location_constraints'] = constraints
    summary['multi_family_sessions'] = conflicts
    summary['constrained_sessions_total'] = sum(bool(r['group_id']) for r in identities)
    (CACHE / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = ['# Exhaustive legacy source-image replay', '',
             'Completed every eligible color image in archive (3) and every static-letter image in archive (5), '
             'using the current importer retry path. All 63 stored float32 coordinates must reproduce within 1e-6, '
             'with the same letter and handedness. No tolerance was widened to increase coverage.', '',
             '| Archive | Images examined | Current rows reproduced |', '|---|---:|---:|']
    for family, s in summary['archives'].items():
        lines.append(f'| {family} | {s["images_processed"]:,} | {s["unique_current_rows_matched"]:,} |')
    lines += ['', f'Added row-level source evidence for **{newly_supported:,}** previously unresolved rows. '
              f'Now **{summary["source_supported_rows_total"]:,} / {len(rows):,}** current rows have import, capture or image-replay source support. '
              f'Added {constraints:,} conservative session co-location constraints; {summary["constrained_sessions_total"]:,} sessions now have a source-family constraint.', '',
              'A replay match establishes compatible stored geometry in a source container. It does not uniquely identify a photo, '
              'prove that a label is linguistically correct, or identify a person. All matching images and subject-folder aliases are retained. '
              'Unmatched rows remain unresolved; an exhaustive pass through these archives with this extractor is not proof that their originals do not exist. '
              'Different historical detector settings, tracking state or preprocessing can prevent exact replay. '
              'A partially matched artificial session receives a conservative source co-location constraint, not an assertion that every row has a verified origin. '
              'Signer IDs and verification status were not fabricated.', '',
              'The original 159,534-row training CSV and existing model weights were unchanged. '
              'See the separate [source-separated training experiment](../../source-separated-training/20261024-source-holdout/report.md) '
              'for the evidence-only subset, all-row exclusion ledger, fixed split and new candidate.', '',
              '[Every image/row match](image-row-matches.csv), [recovery summary and hashes](summary.json), '
              '[updated row provenance](../row-provenance.csv), [updated conservative group plan](../../signer-benchmark/identity-plan.csv).']
    (CACHE / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
