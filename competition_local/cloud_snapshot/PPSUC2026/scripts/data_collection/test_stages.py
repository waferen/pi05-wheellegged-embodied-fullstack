"""Stage configuration and episode snapshot contracts (no simulator needed)."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'source/ppsuc_rl_lab/ppsuc_rl_lab'))
from data_collection.stages import load_stages, STAGES_PATH, STAGE_VERSION, STAGE_PROMPTS
from data_collection.episode import EpisodeWriter


class StageContracts(unittest.TestCase):
    def test_load_outside_project(self):
        before = Path.cwd()
        try:
            os.chdir('/tmp')
            version, stages = load_stages()
            self.assertEqual(version, STAGE_VERSION)
            self.assertEqual(tuple(s['instruction_en'] for s in stages), STAGE_PROMPTS)
        finally:
            os.chdir(before)

    def test_invalid_definitions(self):
        original = json.loads(STAGES_PATH.read_text())
        variants = []
        for change in ('number', 'empty', 'ascii', 'version', 'count'):
            data = copy.deepcopy(original)
            if change == 'number': data['stages'][1]['id'] = 1
            if change == 'empty': data['stages'][0]['instruction_zh'] = ' '
            if change == 'ascii': data['stages'][0]['instruction_en'] = '中文'
            if change == 'version': data['version'] = True
            if change == 'count': data['stages'].pop()
            variants.append(data)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'stages.json'
            for data in variants:
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError): load_stages(path)

    def test_writer_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = EpisodeWriter(tmp)
            try:
                self.assertEqual(writer.file['prompt'].asstr()[:].tolist(), list(STAGE_PROMPTS))
                self.assertEqual(writer.file.attrs['stage_schema_version'], STAGE_VERSION)
            finally:
                writer.close()


if __name__ == '__main__':
    unittest.main()
