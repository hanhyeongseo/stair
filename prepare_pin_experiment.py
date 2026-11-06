"""Create a separate review split without changing original data or labels."""
import copy
import hashlib
import json
from collections import Counter

from mask2former_detector import ROOT


def main():
    baseline=ROOT/'data'/'manifest.json'
    data=json.loads(baseline.read_text(encoding='utf-8'))
    pins=json.loads((ROOT/'outputs'/'review_selection'/'pins.json').read_text(encoding='utf-8'))
    ids={r['id'] for r in data['records']}
    if not pins or not set(pins)<=ids:raise ValueError('Invalid pin selection')
    pinned_groups={r['scene_group'] for r in data['records'] if r['id'] in pins}
    # Fixed before training: keep unpinned original seo validation groups, and
    # the first three unpinned han capture groups for both up/down validation.
    val_groups={r['scene_group'] for r in data['records'] if r['split']=='val' and r['scene_group'] not in pinned_groups}
    han_groups=sorted({r['scene_group'] for r in data['records'] if r['source']=='han' and r['scene_group'] not in pinned_groups})
    val_groups.update(han_groups[:3])
    revised=copy.deepcopy(data)
    for row in revised['records']:
        row['previous_split']=row['split'];row['review_pinned']=row['id'] in pins
        if row['scene_group'] in val_groups:row['split']='val'
        elif row['source']=='park' and row['scene_group'] not in pinned_groups:row['split']='test'
        else:row['split']='train'
    assert all(r['split']=='train' for r in revised['records'] if r['review_pinned'])
    groups={}
    for row in revised['records']:groups.setdefault(row['scene_group'],set()).add(row['split'])
    assert all(len(v)==1 for v in groups.values())
    folder=ROOT/'data'/'experiments'/'pinned_review_v1';folder.mkdir(parents=True,exist_ok=True)
    revised['split_policy']='Review development experiment. All pins and exposure groups enter train. Unpinned han/seo groups select weights. Remaining park photos are same-source development only, NOT independent unseen-site test.'
    revised['baseline_manifest_sha256']=hashlib.sha256(baseline.read_bytes()).hexdigest()
    (folder/'manifest.json').write_text(json.dumps(revised,ensure_ascii=False,indent=2),encoding='utf-8')
    (folder/'pins.json').write_text(json.dumps(pins,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'pinned_train':len(pins),'splits':dict(Counter(r['split'] for r in revised['records'])),
             'validation_groups':sorted(val_groups),'policy':revised['split_policy']}
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
