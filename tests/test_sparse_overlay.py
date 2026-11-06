import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sparse_overlay import SparseMaskTracker


class SparseOverlayTest(unittest.TestCase):
    def test_no_detector_update_tracks_but_fresh_miss_clears(self):
        tracker=SparseMaskTracker();frame=np.full((100,120,3),80,np.uint8)
        mask=np.zeros((100,120),np.uint8);mask[30:70,30:90]=255
        self.assertTrue(tracker.update(frame,mask).any())
        for _ in range(6):self.assertTrue(tracker.update(frame,None).any())
        self.assertFalse(tracker.update(frame,np.zeros_like(mask)).any())
        self.assertFalse(tracker.update(frame,None).any())

    def test_scene_cut_clears_track_without_new_detection(self):
        tracker=SparseMaskTracker();mask=np.zeros((100,120),np.uint8);mask[30:70,30:90]=255
        tracker.update(np.full((100,120,3),20,np.uint8),mask)
        self.assertFalse(tracker.update(np.full((100,120,3),200,np.uint8),None).any())


if __name__=='__main__':unittest.main()
