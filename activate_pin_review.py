"""Activate a checked review model, retaining a recoverable baseline."""
import hashlib
import json
import shutil
import argparse

from mask2former_detector import ROOT


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',default='mask2former_hazard_review_v1')
    parser.add_argument('--experiment',default='pinned_review_v1')
    args=parser.parse_args()
    if not args.candidate.replace('_','').isalnum() or not args.experiment.replace('_','').isalnum():raise ValueError('Invalid experiment')
    report=json.loads((ROOT/'outputs'/args.experiment/'report.json').read_text(encoding='utf-8'))
    if not report['can_activate']:raise RuntimeError('Candidate did not meet the predeclared acceptance rule; baseline retained.')
    candidate=ROOT/'models'/(args.candidate+'.pt')
    active=ROOT/'models/mask2former_hazard_head.pt'
    if digest(candidate)!=report['candidate_sha256'] or digest(active)!=report['baseline_sha256']:
        raise RuntimeError('Model changed since evaluation; refusing to activate a different checkpoint.')
    backup=ROOT/'models/backups/before_pin_review_v1';backup.mkdir(parents=True,exist_ok=True)
    destination=backup/active.name
    if destination.exists() and digest(destination)!=digest(active):raise RuntimeError('A different backup already exists')
    shutil.copy2(active,destination)
    training=ROOT/'models/hazard_head_training.json'
    if training.exists():shutil.copy2(training,backup/training.name)
    temporary=active.with_suffix('.new.pt');shutil.copy2(candidate,temporary);temporary.replace(active)
    shutil.copy2(ROOT/'models'/(args.candidate+'_training.json'),training)
    metadata={'experiment':args.experiment,'checkpoint_sha256':digest(active),'pinned_train_images':44,
              'training_originals':86,'validation_originals':10,'same_source_dev_images':28,
              'summary':report['summary'],'warning':report['warning'],
              'backup':str(destination),'original_manifest_unchanged':True}
    (ROOT/'models/active_hazard_model.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
