import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import activate_pin_review as activation


class ReviewActivationTest(unittest.TestCase):
    def test_rejected_model_cannot_replace_active_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'outputs/pinned_review_v1').mkdir(parents=True);(root/'models').mkdir()
            active=root/'models/mask2former_hazard_head.pt';active.write_bytes(b'baseline')
            (root/'outputs/pinned_review_v1/report.json').write_text(json.dumps({'can_activate':False}))
            with patch.object(activation,'ROOT',root),patch.object(sys,'argv',['activate']):
                with self.assertRaises(RuntimeError):activation.main()
            self.assertEqual(active.read_bytes(),b'baseline')

    def test_acceptance_backs_up_baseline_and_records_exact_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'outputs/pinned_review_v1').mkdir(parents=True);(root/'models').mkdir()
            active=root/'models/mask2former_hazard_head.pt';active.write_bytes(b'baseline')
            candidate=root/'models/mask2former_hazard_review_v1.pt';candidate.write_bytes(b'candidate')
            (root/'models/hazard_head_training.json').write_text('{"old":true}')
            (root/'models/mask2former_hazard_review_v1_training.json').write_text('{"new":true}')
            report={'can_activate':True,'candidate_sha256':hashlib.sha256(b'candidate').hexdigest(),
                    'baseline_sha256':hashlib.sha256(b'baseline').hexdigest(),'summary':{},'warning':'development only'}
            (root/'outputs/pinned_review_v1/report.json').write_text(json.dumps(report))
            with patch.object(activation,'ROOT',root),patch.object(sys,'argv',['activate']):activation.main()
            self.assertEqual(active.read_bytes(),b'candidate')
            self.assertEqual((root/'models/backups/before_pin_review_v1/mask2former_hazard_head.pt').read_bytes(),b'baseline')
            self.assertEqual(json.loads((root/'models/active_hazard_model.json').read_text())['checkpoint_sha256'],report['candidate_sha256'])


if __name__=='__main__':unittest.main()
