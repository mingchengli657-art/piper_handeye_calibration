"""ChArUco target definition shared by detector and collector."""
from pathlib import Path
import cv2
import yaml

DEFAULT_BOARD = Path(__file__).resolve().parents[1] / 'config' / 'board.yaml'


def load_board(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    x, y = int(cfg['squares_x']), int(cfg['squares_y'])
    square, marker = float(cfg['square_length_m']), float(cfg['marker_length_m'])
    if x < 2 or y < 2 or not 0 < marker < square:
        raise ValueError('board requires at least 2x2 squares and 0 < marker < square')
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg['dictionary']))
    if hasattr(cv2.aruco, 'CharucoBoard_create'):
        board = cv2.aruco.CharucoBoard_create(x, y, square, marker, dictionary)
    else:
        board = cv2.aruco.CharucoBoard((x, y), square, marker, dictionary)
        if hasattr(board, 'setLegacyPattern'):
            board.setLegacyPattern(bool(cfg.get('legacy_pattern', True)))
    return cfg, dictionary, board
