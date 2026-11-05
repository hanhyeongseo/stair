"""Fine tune YOLO segmentation; held-out source is never used to choose weights."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.cache'/'ultralytics'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache'/'matplotlib'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--device',default='0')
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--name',default='hazard_seg')
    args=parser.parse_args()
    import cv2
    import torch
    from ultralytics import YOLO
    from vision import enhance
    manifest=json.loads((ROOT/'data'/'manifest.json').read_text(encoding='utf-8'))
    # Exposure variants only in TRAIN, with exactly the same polygon labels.
    train_paths=[]
    for row in manifest['records']:
        if row['split']!='train':continue
        path=Path(row['image']);new=path.with_name(path.stem+'_enh.jpg')
        image=cv2.imread(str(path))
        cv2.imwrite(str(new),enhance(image),[cv2.IMWRITE_JPEG_QUALITY,95])
        label=ROOT/'data'/'yolo'/'labels'/'train'/(path.stem+'.txt')
        shutil.copy2(label,label.with_name(path.stem+'_enh.txt'))
        train_paths.extend([path.as_posix(),new.as_posix()])
    (ROOT/'data'/'yolo'/'train.txt').write_text('\n'.join(train_paths)+'\n',encoding='utf-8')
    model_path=ROOT/'models'/'yolo11n-seg.pt'
    model=YOLO(str(model_path))
    result=model.train(data=str(ROOT/'data'/'hazard.yaml'),epochs=args.epochs,imgsz=args.imgsz,
        batch=8,device=args.device,workers=0,project=str(ROOT/'runs'),name=args.name,
        patience=20,seed=1105,deterministic=True,cache=False,plots=True,
        hsv_h=.025,hsv_s=.45,hsv_v=.55,degrees=8,translate=.1,scale=.35,
        fliplr=.5,flipud=0.,mosaic=.25,close_mosaic=10,mixup=0.,copy_paste=0.,
        optimizer='AdamW',lr0=.001,weight_decay=.0005,amp=True)
    best=Path(model.trainer.best)
    destination=ROOT/'models'/'hazard_best.pt';shutil.copy2(best,destination)
    trained=YOLO(str(destination))
    # Only now evaluate the untouched source. No post-test threshold tuning.
    metrics=trained.val(data=str(ROOT/'data'/'hazard.yaml'),split='test',
        imgsz=args.imgsz,device=args.device,workers=0,project=str(ROOT/'outputs'),
        name=args.name+'_heldout',plots=True)
    report={'model':str(destination),'architecture':'YOLO11n instance segmentation, 1 hazard_step class',
        'train_sources':sorted({r['source'] for r in manifest['records'] if r['split']=='train'}),
        'test_sources':sorted({r['source'] for r in manifest['records'] if r['split']=='test'}),
        'split_policy':manifest['split_policy'],
        'test_metrics':{k:float(v) for k,v in metrics.results_dict.items()},
        'warning':'Held-out capture-source test, not a guarantee for all unseen stairs. No confirmed no-stair negative images were supplied.',
        'manifest_sha256':hashlib.sha256((ROOT/'data'/'manifest.json').read_bytes()).hexdigest(),
        'torch':torch.__version__,'device':args.device,'epochs_requested':args.epochs,
        'epochs_completed':int(model.trainer.epoch)+1}
    (ROOT/'models'/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
