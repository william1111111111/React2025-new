import copy
import json
import unittest
from pathlib import Path
from validate_annotations import validate

EXAMPLES = json.loads(Path(__file__).with_name("examples.synthetic.json").read_text())

class AnnotationTests(unittest.TestCase):
    def test_examples_are_valid_when_explicitly_allowed(self):
        for r in EXAMPLES: validate(r, allow_examples=True)
    def test_examples_blocked_by_default(self):
        with self.assertRaises(ValueError): validate(EXAMPLES[0])
    def test_source_cannot_read_listener(self):
        r=copy.deepcopy(EXAMPLES[0]); r["visible_roles"]=["speaker","listener"]
        with self.assertRaises(ValueError): validate(r,allow_examples=True)
    def test_listener_val_labels_rejected(self):
        r=copy.deepcopy(EXAMPLES[1]); r["split"]="val"
        with self.assertRaises(ValueError): validate(r,allow_examples=True)
    def test_privileged_input_rejected(self):
        r=copy.deepcopy(EXAMPLES[1]); r["allowed_for_model_input"]=True
        with self.assertRaises(ValueError): validate(r,allow_examples=True)
    def test_time_bounds_rejected(self):
        r=copy.deepcopy(EXAMPLES[0]); r["events"][0]["end_s"]=100
        with self.assertRaises(ValueError): validate(r,allow_examples=True)
    def test_event_ids_unique(self):
        r=copy.deepcopy(EXAMPLES[0]); r["events"].append(copy.deepcopy(r["events"][0]))
        with self.assertRaises(ValueError): validate(r,allow_examples=True)
if __name__=="__main__": unittest.main()
