import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hazard_head import HazardHead, probability_mask


class HazardHeadTest(unittest.TestCase):
    def test_high_probability_outside_stairs_does_not_create_detection(self):
        probability = np.ones((24,32),np.float32)
        mask,score = probability_mask(probability,np.zeros((120,160),np.uint8),.5)
        self.assertFalse(mask.any());self.assertIsNone(score)

    def test_component_gate_and_image_alignment(self):
        probability = np.zeros((24,32),np.float32);probability[10:14,5:25] = .9
        region = np.zeros((120,160),np.uint8);region[40:85,20:140]=255
        mask,score = probability_mask(probability,region,.5)
        self.assertEqual(mask.shape,region.shape);self.assertGreater(score,.5)
        self.assertFalse(mask[:35].any());self.assertFalse(mask[90:].any())

    def test_head_uses_frozen_feature_grid_with_bounded_size(self):
        head = HazardHead().eval()
        features = torch.zeros((1,256,128,176))
        with torch.inference_mode():
            output = head(features,torch.ones((1,1,512,704)))
        self.assertEqual(output.shape,(1,1,70,96))


if __name__=='__main__':unittest.main()
