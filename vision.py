"""Low-light preparation and optional color-edge support; no class heuristics."""
import cv2
import numpy as np


def enhance(image):
    clean=cv2.bilateralFilter(image,5,20,5)
    lab=cv2.cvtColor(clean,cv2.COLOR_BGR2LAB)
    light=lab[:,:,0].astype(np.float32)/255
    median=float(np.median(light))
    gamma=float(np.clip(np.log(.4)/np.log(max(.015,min(median,.99))),.4,1))
    lifted=light**gamma
    lifted=np.minimum(lifted,light*6+.015)
    lab[:,:,0]=np.uint8(np.clip(lifted*255,0,255))
    local=cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8)).apply(lab[:,:,0])
    lab[:,:,0]=cv2.addWeighted(lab[:,:,0],.8,local,.2,0)
    return cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)


def color_edges(image,mask):
    """Classroom Lab color difference + Canny inside an already detected region.

    This cannot create or confirm a hazard detection and never moves the mask.
    """
    lab=cv2.cvtColor(cv2.GaussianBlur(image,(5,5),1),cv2.COLOR_BGR2LAB)
    edge=cv2.Canny(lab[:,:,0],40,100)
    color=np.zeros(mask.shape,np.float32)
    for channel in (1,2):
        gx=cv2.Sobel(lab[:,:,channel],cv2.CV_32F,1,0)
        gy=cv2.Sobel(lab[:,:,channel],cv2.CV_32F,0,1)
        color+=cv2.magnitude(gx,gy)
    edge[color>35]=255
    edge=cv2.bitwise_and(edge,mask)
    ys,xs=np.where(mask>0)
    if not len(xs):return edge,[]
    width=int(xs.max()-xs.min())
    raw=cv2.HoughLinesP(edge,1,np.pi/360,30,minLineLength=max(25,int(width*.5)),maxLineGap=8)
    lines=[]
    if raw is not None:
        for x,y,xx,yy in raw[:,0].tolist():
            if x>xx:x,y,xx,yy=xx,yy,x,y
            if abs(np.degrees(np.arctan2(yy-y,xx-x)))>45:continue
            lines.append([x,y,xx,yy])
    lines.sort(key=lambda p:-np.hypot(p[2]-p[0],p[3]-p[1]))
    return edge,lines[:2]
