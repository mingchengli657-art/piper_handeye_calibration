"""Numerical regression, failure cases, and real held-out sample replay."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import validate_transform
from solve_handeye import load_samples, solve, METHODS
from board_config import load_board, DEFAULT_BOARD


class PipelineTest(unittest.TestCase):
    def test_known_transform_recovery(self):
        rng = np.random.default_rng(42)
        def transform():
            t = np.eye(4)
            t[:3, :3] = Rotation.from_rotvec(rng.normal(0, .4, 3)).as_matrix()
            t[:3, 3] = rng.uniform(-.3, .3, 3)
            return t
        expected, fixed_target = transform(), transform()
        rows = []
        for i in range(25):
            base_tcp = transform()
            camera_target = np.linalg.inv(expected) @ np.linalg.inv(base_tcp) @ fixed_target
            rows.append((Path(str(i)), dict(base_T_tcp=base_tcp.tolist(),
                camera_T_target_rvec=Rotation.from_matrix(camera_target[:3, :3]).as_rotvec().tolist(),
                camera_T_target_tvec_m=camera_target[:3, 3].tolist())))
        actual, _ = solve(rows, METHODS['park'])
        np.testing.assert_allclose(actual, expected, atol=1e-8)
        with self.assertRaises(ValueError):
            solve([rows[0]] * 5, METHODS['park'])

    def test_invalid_transform(self):
        for matrix in (np.zeros((4, 4)), np.full((4, 4), np.nan), np.eye(3), np.diag([-1, 1, 1, 1])):
            with self.assertRaises(ValueError):
                validate_transform(matrix)

    def test_bad_samples_filtered(self):
        sample = json.loads((ROOT / 'examples/metadata/train/sample_002.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'sample_001.json').write_text('{broken')
            sample['timestamp_difference_s'] = float('nan')
            (p/'sample_002.json').write_text(json.dumps(sample))
            rows, excluded = load_samples(p, 88, .05)
            self.assertEqual(len(rows), 0)
            self.assertEqual(len(excluded), 2)

    def test_original_board(self):
        cfg, _, board = load_board(DEFAULT_BOARD)
        points = board.chessboardCorners if hasattr(board, 'chessboardCorners') else board.getChessboardCorners()
        self.assertEqual(len(points), 88)
        self.assertEqual(cfg['square_length_m'], .015)

    def test_real_data_cli_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, report = Path(tmp)/'result.yaml', Path(tmp)/'validation.yaml'
            for command in [
                ['solve_handeye.py', '--metadata-dir', str(ROOT/'examples/metadata/train'), '--output', str(result)],
                ['validate_result.py', '--metadata-dir', str(ROOT/'examples/metadata/validation'), '--result', str(result), '--output', str(report)]
            ]:
                subprocess.run([sys.executable, str(ROOT/'scripts'/command[0]), *command[1:]], check=True, capture_output=True, text=True, cwd=tmp)
            actual = yaml.safe_load(result.read_text())
            reference = yaml.safe_load((ROOT/'examples/reference/handeye_result.yaml').read_text())
            self.assertEqual(actual['selection']['used_sample_count'], 43)
            np.testing.assert_allclose(actual['tcp_T_camera']['matrix'], reference['tcp_T_camera']['matrix'], atol=1e-7)
            validation = yaml.safe_load(report.read_text())
            self.assertEqual(validation['sample_count'], 10)
            self.assertEqual(validation['overlap_with_training_ids'], [])
            self.assertAlmostEqual(validation['translation_error_rms_mm'], 1.8611782596, places=5)
            self.assertAlmostEqual(validation['rotation_error_rms_deg'], .18489253095, places=5)


if __name__ == '__main__':
    unittest.main()
