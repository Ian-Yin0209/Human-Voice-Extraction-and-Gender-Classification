import torch
import torchaudio
from librosa.core import load
import numpy as np
import warnings
import soundfile as sf
import wave
import sounddevice as sd
from spleeter_style.estimator import Estimator
def estimate(input,output):
    warnings.filterwarnings('ignore')
    es = Estimator(2, './checkpoints/2stems/model')
    # load wav audio
    wave_file=wave.open(input, 'r')
    channels=wave_file.getnchannels()
    wav, sr = torchaudio.load(input)
    # normalize audio
    wav_torch = wav / (wav.max() + 1e-8)
    wavs = es.separate(wav_torch)
    #print(wav)
    for i in range(len(wavs)):
        file = wavs[i]
        if i ==0 :
            fname = output +'/HumanVoice.wav'
        else:
            fname = output +'/BackGroundMusic.wav'
        print('Writing ',fname)
        sf.write(fname,file[0],sr)
    return output+'/HumanVoice.wav'