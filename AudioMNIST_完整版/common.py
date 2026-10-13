"""Small shared provenance and atomic-output helpers."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 20261009
REPRESENTATIONS = ('logmel', 'mfcc_delta')
MODELS = ('logistic', 'forest', 'lightgbm_goss')
MODES = ('clean_matched', 'augmented')
CONDITIONS = ('clean', 'white_20', 'white_10', 'pink_10')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)
