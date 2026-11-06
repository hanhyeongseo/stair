"""Pretrained stairs backbone and supervised hazard candidate head.

The hazard head learns selected regions; there is no explicit ordinal counting.
"""
import os
import threading
import time
import hashlib
from pathlib import Path

import cv2
import numpy as np

from vision import enhance

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models' / 'mask2former-swin-large-ade-semantic'
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))


class StairsDetector:
    def __init__(self,head_path=None):
        import torch
        from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
        if not (MODEL_DIR / 'model.safetensors').exists():
            raise FileNotFoundError('Mask2Former 가중치 다운로드가 필요합니다.')
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lock = threading.RLock()
        self.processor = Mask2FormerImageProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            str(MODEL_DIR), local_files_only=True, use_safetensors=True).eval().to(self.device)
        self.labels = {int(k): v for k, v in self.model.config.id2label.items()}
        self.stair_ids = [k for k, v in self.labels.items() if v.lower() in ('stairs', 'stairway', 'step')]
        if len(self.stair_ids) != 3:
            raise ValueError('예상한 ADE20K 계단 클래스가 없습니다.')
        self.hazard_head = None
        self.hazard_checkpoint_sha256 = None
        self.hazard_manifest_sha256 = None
        head_path = Path(head_path) if head_path is not None else ROOT/'models'/'mask2former_hazard_head.pt'
        if head_path.exists():
            from hazard_head import HazardHead
            state = torch.load(head_path,map_location=self.device,weights_only=True)
            self.hazard_head = HazardHead().to(self.device).eval()
            self.hazard_head.load_state_dict(state['state_dict'])
            self.hazard_threshold = float(state['threshold'])
            self.hazard_checkpoint_sha256 = hashlib.sha256(head_path.read_bytes()).hexdigest()
            self.hazard_manifest_sha256 = state.get('manifest_sha256')

    def segment(self, image, return_features=False):
        """Official semantic class aggregation followed by per-pixel argmax.

        The normalized semantic scores are relative class support, not calibrated
        probabilities of correct detection. No confidence threshold is invented.
        """
        torch = self.torch
        inputs = self.processor(images=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                                size={'shortest_edge':512, 'longest_edge':768}, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
            semantic = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image.shape[:2]])[0]
        array = semantic.cpu().numpy().astype(np.uint8)
        if return_features:
            return array,outputs.pixel_decoder_last_hidden_state.detach()
        return array

    def infer(self, image, mode='both'):
        if mode not in ('original', 'enhanced', 'both'):
            raise ValueError('입력 방식은 original, enhanced, both 중 하나여야 합니다.')
        start = time.perf_counter()
        h, w = image.shape[:2]
        scale = min(1, 1280 / max(h, w))
        frame = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))))
        corrected = enhance(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        low_signal = float(np.mean(gray < 8)) > .95 or float(gray.std()) < 1.5
        views = {'original': frame, 'enhanced': corrected}
        summaries = {}
        for name in (('original','enhanced') if mode == 'both' else (mode,)):
            source = views[name]
            if low_signal:
                semantic = np.full(frame.shape[:2], 255, np.uint8)
                features = None
            else:
                with self.lock:
                    if getattr(self,'hazard_head',None) is not None:
                        semantic,features = self.segment(source,return_features=True)
                    else:
                        semantic = self.segment(source)
                        features = None
            mask = np.isin(semantic, self.stair_ids).astype(np.uint8)*255
            overlay = source.copy()
            tint = np.full_like(source, (255, 160, 20))
            overlay[mask > 0] = cv2.addWeighted(source, .55, tint, .45, 0)[mask > 0]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 190, 30), 2)
            transparent = np.zeros((*mask.shape, 4), np.uint8)
            transparent[mask > 0] = (255, 160, 20, 110)
            cv2.drawContours(transparent, contours, -1, (255, 190, 30, 255), 2)
            class_areas = {self.labels[k]: int(np.sum(semantic == k)) for k in self.stair_ids}
            summaries[name] = {'status': 'insufficient_signal' if low_signal else 'stairs_region' if np.any(mask) else 'not_detected',
                               'stair_pixels': int(np.count_nonzero(mask)),
                               'stair_area_fraction': float(np.mean(mask > 0)),
                               'class_pixels': class_areas}
            views[name+'_overlay'] = overlay
            views[name+'_mask'] = mask
            views[name+'_transparent'] = transparent
            if features is not None:
                from hazard_head import probability_mask,stair_edges
                region = self.torch.from_numpy((mask>0).astype(np.float32))[None,None].to(self.device)
                with self.lock,self.torch.inference_mode():
                    probability = self.hazard_head(features,region).sigmoid()[0,0].cpu().numpy()
                hazard,score = probability_mask(probability,mask,self.hazard_threshold)
                refined = source.copy()
                lines = stair_edges(source,mask)
                from overlay_style import candidate_overlay,ORANGE
                refined=candidate_overlay(source,hazard)
                hc,_ = cv2.findContours(hazard,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                ht = np.zeros((*mask.shape,4),np.uint8)
                ht[hazard>0] = (*ORANGE,15)
                cv2.drawContours(ht,hc,-1,(*ORANGE,255),2)
                views[name+'_hazard_overlay'] = refined
                views[name+'_hazard_mask'] = hazard
                views[name+'_hazard_transparent'] = ht
                summaries[name]['hazard'] = {'status':'candidate' if score is not None else 'not_detected',
                    'score':score,'threshold':self.hazard_threshold,
                    'score_is_calibrated_probability':False,'pixels':int(np.count_nonzero(hazard)),
                    'edge_candidates':lines,'ordinal_direction_reasoning':False}
        if self.device == 'cuda':
            self.torch.cuda.synchronize()
        report = {'model': 'facebook/mask2former-swin-large-ade-semantic',
                  'task': 'whole_stairs_semantic_segmentation', 'device': self.device,
                  'mode': mode, 'views': summaries, 'input_size': [w,h],
                  'analysis_size': list(frame.shape[1::-1]),
                  'processing_ms': round((time.perf_counter()-start)*1000, 1),
                  'classes': {str(k): self.labels[k] for k in self.stair_ids},
                  'hazard_step_selected': any(v.get('hazard',{}).get('status')=='candidate' for v in summaries.values()),
                  'fine_tuned_on_team_photos': getattr(self,'hazard_head',None) is not None,
                  'backbone_fine_tuned_on_team_photos':False,
                  'hazard_head_trained_on_team_photos':getattr(self,'hazard_head',None) is not None,
                  'hazard_checkpoint_sha256':getattr(self,'hazard_checkpoint_sha256',None),
                  'hazard_training_manifest_sha256':getattr(self,'hazard_manifest_sha256',None),
                  'warning': '주황 윤곽선은 위험 단차 후보, 파랑은 계단 전체입니다. 위험도를 판단해 색을 바꾸지 않습니다. 미검출은 안전 판정이 아닙니다.'}
        return views, report
