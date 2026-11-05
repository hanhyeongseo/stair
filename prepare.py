"""Import user annotations without modifying source files; audit before training."""
import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'data'/'raw'
DATA=ROOT/'data'/'yolo'
SOURCES={
 'han':Path('C:/Users/KDT76/OneDrive/Desktop/계단/Down'),
 'seo':Path('C:/Users/KDT76/OneDrive/Desktop/계단/Up'),
 'park':Path('C:/Users/KDT76/Downloads/박영찬-20261105T072958Z-1-001.zip'),
 'notes':Path('C:/Users/KDT76/Downloads/etc-20261105T072139Z-1-001.zip'),
}


def load_image(path):
    image=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError('image_decode_failed')
    return image


def write_image(path,image):
    ok,buffer=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])
    if not ok:raise ValueError('image_encode_failed')
    buffer.tofile(str(path))


def import_sources():
    for name,source in SOURCES.items():
        target=RAW/name;target.mkdir(parents=True,exist_ok=True)
        if source.is_dir():
            for path in source.rglob('*'):
                if path.is_file() and path.suffix.lower() in ('.jpg','.jpeg','.png','.json','.txt'):
                    dest=target/path.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
                    if not dest.exists():shutil.copy2(path,dest)
        elif source.suffix=='.zip':
            with zipfile.ZipFile(source) as archive:
                for item in archive.infolist():
                    if item.is_dir():continue
                    relative=Path(item.filename.replace('\\','/'))
                    destination=(target/relative).resolve()
                    if not destination.is_relative_to(target.resolve()):raise ValueError('unsafe_zip_path')
                    if relative.suffix.lower() not in ('.jpg','.jpeg','.png','.json','.txt'):continue
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    if not destination.exists():
                        with archive.open(item) as src,destination.open('wb') as dst:shutil.copyfileobj(src,dst)
        else:raise FileNotFoundError(source)


def polygons(path,w,h):
    annotation=path.with_suffix('.json')
    result=[]
    if annotation.exists():
        obj=json.loads(annotation.read_text(encoding='utf-8-sig'))
        if obj.get('imageWidth')!=w or obj.get('imageHeight')!=h:raise ValueError('annotation_image_size_mismatch')
        for shape in obj.get('shapes',[]):
            points=np.array(shape.get('points',[]),dtype=float)
            kind=shape.get('shape_type','polygon')
            if kind=='rectangle' and points.shape==(2,2):
                (x,y),(xx,yy)=points;points=np.array([[x,y],[xx,y],[xx,yy],[x,yy]])
            elif kind!='polygon':raise ValueError('unsupported_shape_'+kind)
            result.append({'source_class':shape.get('label','unknown'),'points':(points/[w,h]).tolist()})
    else:
        candidates=[path.with_suffix('.txt'),path.parent/'labels'/(path.stem+'.txt')]
        txt=next((p for p in candidates if p.exists()),None)
        if txt is None:return [],None
        annotation=txt
        for line in txt.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            values=line.split()
            if len(values)<7 or (len(values)-1)%2:raise ValueError('expected_yolo_polygon_not_box')
            result.append({'source_class':values[0],'points':np.array(values[1:],float).reshape(-1,2).tolist()})
    for poly in result:
        pts=np.array(poly['points'],dtype=float)
        if pts.ndim!=2 or pts.shape[1]!=2 or len(pts)<3 or not np.isfinite(pts).all():raise ValueError('invalid_polygon')
        if pts.min()<-.002 or pts.max()>1.002:raise ValueError('polygon_out_of_bounds')
        pts=np.clip(pts,0,.999999)
        if abs(cv2.contourArea(pts.astype(np.float32)))<.00001:raise ValueError('polygon_area_too_small')
        poly['points']=pts.tolist()
    return result,str(annotation)


def pair_id(source,stem):
    # Every exposure and extra shot of a numbered view stays together.
    match=re.fullmatch(r'((?:han|seo)\d+)_([01])(?:_\d+)?',stem)
    return source+'/'+(match[1] if match else stem)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--import-sources',action='store_true')
    args=parser.parse_args()
    if args.import_sources:import_sources()
    DATA.mkdir(parents=True,exist_ok=True);(ROOT/'outputs').mkdir(exist_ok=True)
    rows=[];excluded=[];hashes={}
    # Park is a completely held-out capture source. Its labels never select epochs.
    # Within local sources, pair-group validation only selects a checkpoint.
    config=ROOT/'scene_groups.json'
    mapping=json.loads(config.read_text(encoding='utf-8')) if config.exists() else {}
    for source in ('han','seo','park'):
        for path in sorted((RAW/source).rglob('*')):
            if path.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
            try:
                image=load_image(path);h,w=image.shape[:2]
                shapes,annotation=polygons(path,w,h)
                if not shapes:
                    excluded.append({'source':source,'image':str(path),'reason':'unlabeled_or_empty_not_a_confirmed_negative'});continue
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in hashes:
                    excluded.append({'image':str(path),'reason':'exact_duplicate','original':hashes[digest]});continue
                hashes[digest]=str(path)
                group=pair_id(source,path.stem)
                scene=mapping.get(group,group)
                if source=='park':split='test'
                else:
                    bucket=int(hashlib.sha256(scene.encode()).hexdigest()[:8],16)%5
                    split='val' if bucket==0 else 'train'
                name=source+'_'+hashlib.sha256(str(path.relative_to(RAW)).encode()).hexdigest()[:12]
                out_image=DATA/'images'/split/(name+'.jpg');out_label=DATA/'labels'/split/(name+'.txt')
                out_image.parent.mkdir(parents=True,exist_ok=True);out_label.parent.mkdir(parents=True,exist_ok=True)
                scale=min(1,1280/max(w,h));small=cv2.resize(image,(round(w*scale),round(h*scale)))
                write_image(out_image,small)
                text=['0 '+' '.join(f'{v:.7f}' for point in shape['points'] for v in point) for shape in shapes]
                out_label.write_text('\n'.join(text)+'\n',encoding='utf-8')
                gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
                rows.append({'id':name,'source':source,'original':str(path),'image':str(out_image),
                    'annotation':annotation,'split':split,'group':group,'scene_group':scene,'sha256':digest,
                    'polygons':shapes,'mean_gray':float(gray.mean()),'size':[w,h]})
            except (ValueError,OSError,json.JSONDecodeError) as exc:
                excluded.append({'image':str(path),'reason':str(exc)})
    # A physical scene override must not appear in both the held-out source and train.
    seen={}
    for row in rows:
        seen.setdefault(row['scene_group'],set()).add(row['split'])
    bad={k:sorted(v) for k,v in seen.items() if len(v)>1}
    if bad:raise ValueError('scene_split_leakage: '+str(bad))
    manifest={'classes':{'0':'hazard_step'},'label_policy':'Preserve human selected polygon; Down=second step, Up=first step. Merge original class IDs into a hazard-region task.',
        'split_policy':'park capture source held out as test; local validation grouped by exposure pair, NOT guaranteed unseen-location validation.',
        'records':rows,'excluded':excluded,'scene_overrides':mapping}
    (ROOT/'data'/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    counts={s:sum(r['split']==s for r in rows) for s in ('train','val','test')}
    if not all(counts.values()):raise ValueError('empty_split: '+str(counts))
    # Explicit file lists prevent leftovers from earlier dataset versions leaking
    # into a new split, without deleting user files or entire directories.
    for split in ('train','val','test'):
        (DATA/(split+'.txt')).write_text('\n'.join(Path(r['image']).as_posix() for r in rows if r['split']==split)+'\n',encoding='utf-8')
    yaml='path: '+DATA.as_posix()+'\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: hazard_step\n'
    (ROOT/'data'/'hazard.yaml').write_text(yaml,encoding='utf-8')
    # Label overlays make mistakes visible before any metric is trusted.
    for source in ('han','seo','park'):
        selected=[r for r in rows if r['source']==source][:12];tiles=[]
        for row in selected:
            frame=load_image(Path(row['image']));fh,fw=frame.shape[:2]
            for poly in row['polygons']:
                pts=np.round(np.array(poly['points'])*[fw,fh]).astype(np.int32)
                cv2.polylines(frame,[pts],True,(0,255,255),max(2,fw//300))
            frame=cv2.resize(frame,(225,300));cv2.putText(frame,Path(row['original']).stem[:22],(5,20),0,.4,(0,255,0),1)
            tiles.append(frame)
        if tiles:
            while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
            sheet=np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])
            write_image(ROOT/'outputs'/(source+'_labels.jpg'),sheet)
    print(json.dumps({'splits':counts,'excluded':len(excluded),'sources':{s:sum(r['source']==s for r in rows) for s in ('han','seo','park')}},indent=2))


if __name__=='__main__':main()
