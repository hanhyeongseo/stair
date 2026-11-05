"""Freeze Mask2Former, train a small hazard head on train only, select on val.

Existing park test has been inspected in prior work; later evaluations are
development comparisons, not a fresh independent unseen-site benchmark.
"""
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from hazard_head import HazardHead, compact_features, probability_mask
from mask2former_detector import ROOT, StairsDetector
from prepare import load_image
from vision import enhance


def main():
    torch.manual_seed(1105);np.random.seed(1105);random.seed(1105)
    manifest = ROOT/'data'/'manifest.json'
    records = json.loads(manifest.read_text(encoding='utf-8'))['records']
    cache = ROOT/'.cache'/'hazard_features_v1'/hashlib.sha256(manifest.read_bytes()).hexdigest()[:16]
    cache.mkdir(parents=True,exist_ok=True)
    net = StairsDetector()
    device = net.device
    splits = {'train':[],'val':[]}
    for index,row in enumerate(records):
        if row['split'] not in splits:continue
        image = load_image(Path(row['image']))
        h,w = image.shape[:2];scale = min(1,1280/max(h,w))
        image = cv2.resize(image,(round(w*scale),round(h*scale)))
        h,w = image.shape[:2]
        truth = np.zeros((h,w),np.uint8)
        for polygon in row['polygons']:
            cv2.fillPoly(truth,[np.round(np.array(polygon['points'])*[w,h]).astype(np.int32)],1)
        for name,view in [('original',image),('enhanced',enhance(image))]:
            path = cache/(row['id']+'_'+name+'.pt')
            needs_full = row['split']=='val' and path.exists() and 'target_full' not in torch.load(path,map_location='cpu',weights_only=True)
            if not path.exists() or needs_full:
                semantic,features = net.segment(view,return_features=True)
                features = compact_features(features).half().cpu()
                hh,ww = features.shape[-2:]
                region = cv2.resize(np.isin(semantic,net.stair_ids).astype(np.uint8),(ww,hh),interpolation=cv2.INTER_NEAREST)
                target = cv2.resize(truth,(ww,hh),interpolation=cv2.INTER_NEAREST)
                sample = {'features':features,'region':torch.from_numpy(region)[None,None],
                          'target':torch.from_numpy(target)[None,None]}
                if row['split']=='val':
                    sample['target_full'] = torch.from_numpy(truth)
                    sample['region_full'] = torch.from_numpy(np.isin(semantic,net.stair_ids).astype(np.uint8))
                torch.save(sample,path)
            splits[row['split']].append(path)
        print(f'features {index+1}/{len(records)} {row["id"]}',flush=True)
    # Release the frozen backbone before training the head.
    del net
    if device=='cuda':torch.cuda.empty_cache()
    data = {key:[torch.load(path,map_location='cpu',weights_only=True) for path in paths] for key,paths in splits.items()}
    head = HazardHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(),lr=.0005,weight_decay=.01)
    best = -1;stale = 0;history = []
    checkpoint = ROOT/'models'/'mask2former_hazard_head.pt'
    for epoch in range(1,61):
        head.train();random.shuffle(data['train']);losses = []
        for sample in data['train']:
            features = sample['features'].to(device).float()
            region = sample['region'].to(device).float();target = sample['target'].to(device).float()
            if random.random()<.5:
                features=features.flip(-1);region=region.flip(-1);target=target.flip(-1)
            logits = head(features,region)
            bce = F.binary_cross_entropy_with_logits(logits,target,pos_weight=torch.tensor(5.,device=device))
            p = logits.sigmoid()
            dice = 1-(2*(p*target).sum()+1)/(p.sum()+target.sum()+1)
            loss = bce+dice
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        head.eval();values = {t:[] for t in (.25,.35,.45,.55,.65,.75)}
        with torch.inference_mode():
            for sample in data['val']:
                probability = head(sample['features'].to(device),sample['region'].to(device)).sigmoid()[0,0].cpu().numpy()
                region = sample['region_full'].numpy().astype(np.uint8)*255
                target = sample['target_full'].numpy()>0
                for threshold in values:
                    mask,_ = probability_mask(probability,region,threshold)
                    intersection = np.logical_and(mask>0,target).sum()
                    union = np.logical_or(mask>0,target).sum()
                    values[threshold].append(float(intersection/max(1,union)))
        means = {t:float(np.mean(v)) for t,v in values.items()}
        threshold = max(means,key=means.get);score = means[threshold]
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'val_mean_mask_iou':score,'threshold':threshold})
        print(json.dumps(history[-1]),flush=True)
        if score>best+1e-4:
            best=score;stale=0
            torch.save({'state_dict':{k:v.cpu() for k,v in head.state_dict().items()},
                        'threshold':threshold,'epoch':epoch,'val_mean_mask_iou':score,
                        'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()},checkpoint)
        else:stale+=1
        if stale>=12:break
    report = {'architecture':'Frozen Mask2Former pixel decoder + supervised convolutional hazard head',
              'train_originals':len(splits['train'])//2,'val_originals':len(splits['val'])//2,
              'test_used_for_training_or_selection':False,
              'label_policy':'Human-selected Down second / Up first hazard polygon; no explicit direction or ordinal counting.',
              'whole_stairs_labels_available':False,'whole_stairs_backbone_finetuned':False,
              'best_val_mean_mask_iou':best,'history':history}
    (ROOT/'models'/'hazard_head_training.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
