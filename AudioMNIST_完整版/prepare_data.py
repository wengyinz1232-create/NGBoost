"""Rebuild all features from source WAVs; never import old caches or results."""
import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from common import ROOT, SEED, REPRESENTATIONS, CONDITIONS, sha, save_json
from features import audio_features, noisy_audio, feature_names

REVISION = '630d7dab4c040882834de6fa21baf9a60372accd'
URL = f'https://codeload.github.com/soerenab/AudioMNIST/zip/{REVISION}'


def prepare(download=False):
    archive_path = ROOT/'data_raw'/'audiomnist.zip'
    if not archive_path.exists():
        if not download:
            raise FileNotFoundError('Put the source archive in data_raw/audiomnist.zip, or use --download.')
        archive_path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, archive_path.with_suffix('.download'))
        archive_path.with_suffix('.download').replace(archive_path)
    signature = {'raw_sha256': sha(archive_path),
                 'frontend_sha256': sha(ROOT/'features.py'),
                 'prepare_sha256': sha(Path(__file__))}
    inventory_path = ROOT/'data'/'inventory.json'
    if inventory_path.exists():
        old = json.loads(inventory_path.read_text(encoding='utf-8'))
        if old['signature'] != signature:
            raise ValueError('Source/code changed: use a new folder; do not mix old and new features.')
        if all(sha(ROOT/'data'/f'{r}.npz') == old['cache_sha256'][r] for r in REPRESENTATIONS):
            print('DATA_ALREADY_VERIFIED', flush=True); return
        raise ValueError('Feature cache hash mismatch')
    (ROOT/'data').mkdir(exist_ok=True)
    matrices = {r:{c:[] for c in CONDITIONS} for r in REPRESENTATIONS}
    ys, groups, ids, durations, pcm_hashes = [], [], [], [], []
    with zipfile.ZipFile(archive_path) as archive:
        revision = archive.comment.decode('utf-8')
        if revision != REVISION:
            raise ValueError(f'Unexpected archive revision: {revision}')
        files = sorted(n for n in archive.namelist() if n.endswith('.wav') and '/data/' in n)
        assert len(files) == 30000
        selected = [n for n in files if int(Path(n).stem.split('_')[-1]) < 20]
        assert len(selected) == 12000
        # Keep official licence and dataset reference for teammates.
        for base in ('LICENSE','README.md'):
            name = next(n for n in archive.namelist() if n.count('/') == 1 and n.endswith('/'+base))
            (ROOT/'data_raw'/('SOURCE_'+base)).write_bytes(archive.read(name))
        for start in range(0,len(selected),64):
            batch = []
            for name in selected[start:start+64]:
                rate,x = wavfile.read(io.BytesIO(archive.read(name)))
                assert x.ndim == 1 and x.dtype == np.int16 and rate == 48000
                durations.append(len(x)/rate)
                pcm_hashes.append(hashlib.sha256(x.tobytes()).hexdigest())
                divisor = np.gcd(rate,16000)
                x = resample_poly(x.astype(np.float32)/32768,16000//divisor,rate//divisor)
                assert len(x) <= 16000
                batch.append(np.pad(x,(0,16000-len(x))))
                digit,speaker,_ = Path(name).stem.split('_')
                ys.append(int(digit)); groups.append(int(speaker)); ids.append(name)
            waves = np.stack(batch)
            variants = {'clean':waves, 'white_20':noisy_audio(waves,20,SEED+start),
                        'white_10':noisy_audio(waves,10,SEED+start),
                        'pink_10':noisy_audio(waves,10,SEED+start,pink=True)}
            for condition,w in variants.items():
                for rep,x in audio_features(w).items(): matrices[rep][condition].append(x)
            if start % 1024 == 0: print('FEATURES',start,'/ 12000',flush=True)
    assert len(set(ids)) == len(set(pcm_hashes)) == 12000
    names = {}
    for rep,variants in matrices.items():
        variants = {c:np.concatenate(x) for c,x in variants.items()}
        names[rep] = feature_names(rep)
        for x in variants.values():
            assert x.shape == (12000,len(names[rep])) and np.isfinite(x).all()
        np.savez_compressed(ROOT/'data'/f'{rep}.npz',y=np.array(ys),groups=np.array(groups),
                            ids=np.array(ids),**variants)
    save_json(ROOT/'data'/'feature_names.json',names)
    save_json(ROOT/'data'/'sample_manifest.json',{'ids':ids,'y':ys,'speaker_ids':groups,'pcm_sha256':pcm_hashes})
    save_json(inventory_path,{'signature':signature,'revision':revision,'source_url':URL,
        'original_recordings':30000,'selected_recordings':12000,'speakers':60,'classes':10,
        'selection':'Every speaker, every digit, recording indices 0 through 19; same subset as exploration.',
        'class_counts':dict(Counter(ys)),'original_duration_range_seconds':[min(durations),max(durations)],
        'duplicate_pcm_count':0,'feature_dimensions':{r:len(v) for r,v in names.items()},
        'conditions':CONDITIONS,'seed':SEED,'cache_sha256':{r:sha(ROOT/'data'/f'{r}.npz') for r in REPRESENTATIONS}})
    print('PREPARE_COMPLETE: 12000 independent recordings, 256/320 features',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--download',action='store_true')
    prepare(p.parse_args().download)
