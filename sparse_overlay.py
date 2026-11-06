"""Track display geometry between sparse detector updates; no new detections."""
import cv2
import numpy as np


class SparseMaskTracker:
    def __init__(self):
        self.gray=None;self.mask=None

    def update(self,image,detection=None):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        h,w=gray.shape;scale=min(1,320/max(h,w))
        small=cv2.resize(gray,(round(w*scale),round(h*scale)))
        if detection is not None:
            # A fresh miss clears the previous detection immediately.
            current=(detection>0).astype(np.float32)
        elif self.mask is None or self.gray is None:
            current=np.zeros((h,w),np.float32)
        elif np.mean(cv2.absdiff(small,self.gray))>35:
            current=np.zeros((h,w),np.float32)
        else:
            flow=cv2.calcOpticalFlowFarneback(small,self.gray,None,.5,3,21,3,5,1.2,0)
            flow=cv2.resize(flow,(w,h));flow[:,:,0]*=w/small.shape[1];flow[:,:,1]*=h/small.shape[0]
            xx,yy=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
            current=cv2.remap(self.mask,xx+flow[:,:,0],yy+flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        self.gray=small;self.mask=current
        return (current>=.5).astype(np.uint8)*255
