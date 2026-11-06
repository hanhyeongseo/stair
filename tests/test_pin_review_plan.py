import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prepare_pin_experiment as plan


class ReviewSplitTest(unittest.TestCase):
    def test_pins_and_exposure_siblings_enter_train_original_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'data').mkdir();(root/'outputs/review_selection').mkdir(parents=True)
            rows=[]
            for source,group,split in [('han','one','val'),('han','two','train'),('han','three','train'),('han','four','train'),('seo','five','val'),('park','six','test'),('park','seven','test')]:
                for exposure in ('0','1'):
                    rows.append({'id':source+group+exposure,'source':source,'scene_group':source+'/'+group,'split':split})
            manifest=root/'data/manifest.json';manifest.write_text(json.dumps({'records':rows}))
            original=manifest.read_bytes();pins=['hanone0','parksix0']
            (root/'outputs/review_selection/pins.json').write_text(json.dumps(pins))
            with patch.object(plan,'ROOT',root):plan.main()
            revised=json.loads((root/'data/experiments/pinned_review_v1/manifest.json').read_text())
            self.assertEqual(manifest.read_bytes(),original)
            self.assertTrue(all(r['split']=='train' for r in revised['records'] if r['scene_group'] in ('han/one','park/six')))
            self.assertTrue(all(r['split']=='test' for r in revised['records'] if r['scene_group']=='park/seven'))
            groups={}
            for row in revised['records']:groups.setdefault(row['scene_group'],set()).add(row['split'])
            self.assertTrue(all(len(v)==1 for v in groups.values()))


if __name__=='__main__':unittest.main()
