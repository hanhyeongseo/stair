"""Neutral orange candidate outline; no inferred danger-level color changes."""
import cv2
import numpy as np

ORANGE=(0,165,255)


def candidate_overlay(image,mask):
    overlay=image.copy()
    # Very faint fill keeps the step surface visible.
    tinted=cv2.addWeighted(image,.94,np.full_like(image,ORANGE),.06,0)
    overlay[mask>0]=tinted[mask>0]
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay,contours,-1,ORANGE,2,cv2.LINE_AA)
    return overlay


class MotionMaskSmoother:
    """Warp previous support, smooth weakly, confirm abrupt candidate switches.

    Up to two missing frames may be bridged only inside current stairs support.
    This is display postprocessing, not an improvement in raw model accuracy.
    """
    def __init__(self):
        self.previous_gray=None;self.previous=None;self.missing=0
        self.pending=None;self.pending_count=0

    def update(self,image,raw_mask,support):
        h,w=raw_mask.shape
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        scale=min(1,320/max(h,w));small=cv2.resize(gray,(max(2,round(w*scale)),max(2,round(h*scale))))
        current=(raw_mask>0).astype(np.float32)
        gate=cv2.dilate((support>0).astype(np.uint8),np.ones((9,9),np.uint8))
        cut=self.previous_gray is not None and (self.previous_gray.shape!=small.shape or np.mean(cv2.absdiff(small,self.previous_gray))>35)
        if self.previous is None or cut:
            smoothed=current.copy();self.missing=0;self.pending=None;self.pending_count=0
        else:
            flow=cv2.calcOpticalFlowFarneback(small,self.previous_gray,None,.5,3,21,3,5,1.2,0)
            flow=cv2.resize(flow,(w,h));flow[:,:,0]*=w/small.shape[1];flow[:,:,1]*=h/small.shape[0]
            xx,yy=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
            warped=cv2.remap(self.previous,xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            if not current.any():
                self.missing+=1
                smoothed=warped*.8 if self.missing<=2 else np.zeros_like(current)
                self.pending=None;self.pending_count=0
            else:
                self.missing=0
                a=warped>=.5;b=current>0
                overlap=np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum())
                if a.any() and overlap<.10:
                    consistent=self.pending is not None and np.logical_and(self.pending,b).sum()/max(1,np.logical_or(self.pending,b).sum())>.25
                    self.pending_count=self.pending_count+1 if consistent else 1;self.pending=b.copy()
                    # Briefly abstain during a switch rather than keeping an old
                    # region at an unrelated screen location.
                    smoothed=current.copy() if self.pending_count>=3 else np.zeros_like(current)
                else:
                    smoothed=.45*current+.55*warped
                    self.pending=None;self.pending_count=0
        smoothed*=gate
        self.previous=smoothed.astype(np.float32);self.previous_gray=small
        return (smoothed>=.5).astype(np.uint8)*255
