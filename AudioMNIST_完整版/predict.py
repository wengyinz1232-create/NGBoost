"""Predict a <=1-second mono WAV with a trusted saved model; not an evaluation."""
import argparse
import json
import pickle
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from features import audio_features
from evaluation import transform, proba
from common import ROOT, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--wav',required=True)
    args=p.parse_args()
    # Pickle can execute code: only use your own or trusted model artifacts.
    with open(args.model,'rb') as stream:bundle=pickle.load(stream)
    if bundle['source_signature'] != sha(ROOT/'features.py'):
        raise ValueError('Feature code differs from the saved model; use its matching frontend')
    rate,w=wavfile.read(args.wav)
    if w.ndim!=1:raise ValueError('A mono recording is required')
    if w.dtype==np.int16:w=w.astype(np.float32)/32768
    elif np.issubdtype(w.dtype,np.floating):w=w.astype(np.float32)
    else:raise ValueError('Use 16-bit PCM or floating-point WAV')
    g=np.gcd(rate,16000);w=resample_poly(w,16000//g,rate//g)
    if len(w)>16000:raise ValueError('Recording must be no longer than 1 second; no silent cropping')
    w=np.pad(w,(0,16000-len(w)))
    x=audio_features(w[None,:])[bundle['representation']]
    probabilities=proba(bundle['estimator'],transform(x,bundle['scaler']),bundle['classes'])[0]
    print(json.dumps({'prediction':int(bundle['classes'][np.argmax(probabilities)]),
        'probabilities':{str(c):float(v) for c,v in zip(bundle['classes'],probabilities)},
        'note':'A single prediction is not a held-out evaluation result.'},indent=2))


if __name__=='__main__':main()
