import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from overlay_style import candidate_overlay,MotionMaskSmoother


class OverlayTest(unittest.TestCase):
    def test_orange_outline_preserves_input_and_faint_interior(self):
        image=np.full((100,120,3),80,np.uint8);source=image.copy()
        mask=np.zeros((100,120),np.uint8);mask[20:70,20:100]=255
        out=candidate_overlay(image,mask)
        np.testing.assert_array_equal(image,source)
        np.testing.assert_array_equal(out[0,0],image[0,0])
        self.assertLess(int(out[40,50,2])-80,15)
        self.assertGreater(out[20,50,1],100)

    def test_missing_prediction_expires_after_two_frames(self):
        smoother=MotionMaskSmoother();image=np.full((100,120,3),80,np.uint8)
        mask=np.zeros((100,120),np.uint8);mask[30:70,30:90]=255
        gate=np.full(mask.shape,255,np.uint8)
        self.assertTrue(smoother.update(image,mask,gate).any())
        blank=np.zeros_like(mask)
        self.assertTrue(smoother.update(image,blank,gate).any())
        self.assertTrue(smoother.update(image,blank,gate).any())
        self.assertFalse(smoother.update(image,blank,gate).any())

    def test_no_current_stairs_support_immediately_clears_display(self):
        smoother=MotionMaskSmoother();image=np.full((100,120,3),80,np.uint8)
        mask=np.zeros((100,120),np.uint8);mask[30:70,30:90]=255
        smoother.update(image,mask,np.full_like(mask,255))
        self.assertFalse(smoother.update(image,np.zeros_like(mask),np.zeros_like(mask)).any())


if __name__=='__main__':unittest.main()
