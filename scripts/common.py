"""Shared input validation for offline solve and validation tools."""
import numpy as np


def validate_transform(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('transform must be a finite 4x4 matrix')
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError('invalid homogeneous bottom row')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError('rotation must be orthonormal with determinant +1')
    return matrix


def validate_sample(data):
    int(data['sample_id'])
    validate_transform(data['base_T_tcp'])
    for key in ('camera_T_target_rvec', 'camera_T_target_tvec_m'):
        value = np.asarray(data[key], dtype=float)
        if value.size != 3 or not np.isfinite(value).all():
            raise ValueError(f'{key} must contain 3 finite numbers')
    dt = float(data['timestamp_difference_s'])
    if not np.isfinite(dt) or dt < 0:
        raise ValueError('timestamp difference must be finite and nonnegative')
