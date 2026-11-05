"""Small supervised hazard-step decoder on frozen Mask2Former features."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def compact_features(features):
    h,w = features.shape[-2:]
    if max(h,w)>96:
        scale = 96/max(h,w)
        features = F.interpolate(features.float(),size=(round(h*scale),round(w*scale)),mode='bilinear',align_corners=False)
    return features


class HazardHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(259,64,3,padding=1), nn.GroupNorm(8,64), nn.GELU(),
            nn.Dropout2d(.15), nn.Conv2d(64,32,3,padding=2,dilation=2),
            nn.GroupNorm(8,32), nn.GELU(), nn.Conv2d(32,1,1))

    def forward(self, features, region):
        features = compact_features(features)
        region = F.interpolate(region.float(),size=features.shape[-2:],mode='nearest')
        h,w = features.shape[-2:]
        y,x = torch.meshgrid(torch.linspace(-1,1,h,device=features.device),
                             torch.linspace(-1,1,w,device=features.device),indexing='ij')
        coords = torch.stack([x,y])[None].expand(features.shape[0],-1,-1,-1)
        return self.layers(torch.cat([features.float(),region,coords],dim=1))


def probability_mask(probability, region, threshold):
    """Gate by the model region, never by ground-truth annotations."""
    h,w = region.shape
    resized = cv2.resize(probability.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
    radius = max(1,round(min(h,w)*.004))
    gate = cv2.dilate((region>0).astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8))
    binary = ((resized>=threshold)&(gate>0)).astype(np.uint8)
    count,labels,stats,_ = cv2.connectedComponentsWithStats(binary,8)
    components = []
    for label in range(1,count):
        area = int(stats[label,cv2.CC_STAT_AREA])
        if not .001*h*w <= area <= .5*h*w: continue
        mask = labels==label
        score = float(resized[mask].mean())
        components.append((score,mask))
    if not components: return np.zeros((h,w),np.uint8),None
    score,mask = max(components,key=lambda item:item[0])
    return mask.astype(np.uint8)*255,score


def stair_edges(image,region):
    """Visible line candidates, not confirmed step boundaries or a step count."""
    from vision import enhance
    gray = cv2.cvtColor(enhance(image),cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),1)
    edges = cv2.Canny(gray,35,100)
    edges[region==0] = 0
    h,w = region.shape
    raw = cv2.HoughLinesP(edges,1,np.pi/360,max(20,w//24),
                         minLineLength=max(25,w//7),maxLineGap=max(8,w//60))
    candidates = []
    if raw is not None:
        for x,y,xx,yy in raw[:,0].tolist():
            if x>xx: x,y,xx,yy = xx,yy,x,y
            if abs(np.degrees(np.arctan2(yy-y,xx-x)))>50: continue
            candidates.append([x,y,xx,yy])
    candidates.sort(key=lambda p:-np.hypot(p[2]-p[0],p[3]-p[1]))
    selected = []
    for line in candidates:
        mid = (line[1]+line[3])/2
        if any(abs(mid-(p[1]+p[3])/2)<max(5,h*.008) and
               abs(np.arctan2(line[3]-line[1],line[2]-line[0])-np.arctan2(p[3]-p[1],p[2]-p[0]))<.12 for p in selected): continue
        selected.append(line)
        if len(selected)>=24: break
    return selected
