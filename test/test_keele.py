import pitch_squeezer as ps
import librosa
import pyworld as pw
import numpy as np
import time
import matplotlib.pyplot as plt
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
import parselmouth
import soundfile as sf
import os
from pygame import mixer

def _highlight(f0s, index):
    for line in plt.gca().lines:
        line.remove()
    colors = ["green", "blue", "red", "purple"]
    for i in range(len(f0s)):
        f0s[i][(f0s[i]==0)] = np.nan
        if i == index:
            plt.plot(f0s[i], color=colors[i], linewidth=2)
        else:
            plt.plot(f0s[i], color=colors[i],alpha=0.15)
        plt.pause(0.1)

def play(wav):
    
    def play_wav(wav):
        mixer.music.load(wav)
        mixer.music.play()
   
    play_wav(wav)
    input("press ENTER to stop playback")
    mixer.music.stop()
   
   
def vde(test_f0, ref_f0):
    
    test_voiced = test_f0 > 0
    ref_voiced = ref_f0 > 0

    # Step 3: Calculate Voicing Decision Error
    TP = np.sum(np.logical_and(test_voiced, ref_voiced))
    TN = np.sum(np.logical_and(~test_voiced, ~ref_voiced))
    FP = np.sum(np.logical_and(test_voiced, ~ref_voiced))
    FN = np.sum(np.logical_and(~test_voiced, ref_voiced))

    # Step 4: Calculate Voicing Error Metrics
    VDE = (FP + FN) / (TP + TN + FP + FN)
    return VDE
def mae(test_f0, ref_f0):
    test_voiced = test_f0 > 0
    ref_voiced = ref_f0 > 0
    voiced_frames = np.logical_and(test_voiced, ref_voiced)

    return np.mean(np.abs(test_f0[voiced_frames]-ref_f0[voiced_frames]))
   
def gross(test_f0, ref_f0, voicing_threshold=0, deviation_percentage=20):
  
    test_voiced = test_f0 > voicing_threshold
    ref_voiced = ref_f0 > voicing_threshold

   
    voiced_frames = np.logical_and(test_voiced, ref_voiced)

    # Step 3: Identify frames with significant pitch deviation (>20%)
    deviation_threshold = deviation_percentage / 100.0
    pitch_deviation = np.abs(test_f0[voiced_frames] - ref_f0[voiced_frames]) / ref_f0[voiced_frames]
    
    gross_errors = np.sum(pitch_deviation > deviation_threshold) / np.sum(voiced_frames)
   
    return gross_errors*100

def analysis(wav, fmin, fmax, thresh):
    

  
    f0_ps, if0_ps = ps.track_pitch(wav, voicing_thresh=thresh, min_hz=fmin, max_hz=fmax,target_rate=100)
  
    return f0_ps
    signal = basic.SignalObj(wav)
    f0_pyaapt = pYAAPT.yaapt(signal, **{'f0_min' : fmin, 'f0_max':fmax, 'frame_space' : 10.0})
    f0_pyaapt = f0_pyaapt.samp_values
    return f0_pyaapt
    
    x, fs = librosa.load(wav, sr=None)
    f0_pyin, voiced_flag, voiced_probs = librosa.pyin(x,sr=fs, fmin=fmin, fmax=fmax, hop_length = int(fs/100))
  
    return(f0_pyin)
    ref[ref==0] = np.nan

    ref[ref<120] = np.nan
    f0_ps[f0_ps==0] = np.nan
    plt.plot(f0_ps, label="squeezer")
    #plt.plot(f0_pyin, label="pyin")
    plt.plot(ref, label="ref")
    plt.legend()
    plt.show()
    return
    start_time = time.time()
    x, fs = librosa.load(wav, sr=None)
    f0_pyin, voiced_flag, voiced_probs = librosa.pyin(x,sr=fs, fmin=fmin, fmax=fmax, hop_length = int(fs/200))
    print("pyin analysis done in ", time.time()-start_time, "seconds")
    start_time = time.time()

    signal = basic.SignalObj(wav)
    f0_pyaapt = pYAAPT.yaapt(signal, **{'f0_min' : fmin, 'f0_max':fmax, 'frame_space' : 5.0})
    f0_pyaapt = f0_pyaapt.samp_values
   
    print("pYAAPT analysis done in ", time.time()-start_time, "seconds")


  
    
if __name__ == "__main__":
    import sys, glob
    test_files = sorted(glob.glob(sys.argv[1]+"/*/signal.wav"))
    ref_files = sorted(glob.glob(sys.argv[1]+"/*/*.npy"))

    import random
    #random.shuffle(test_files)
    plt.figure(figsize=(12,4))
    plt.ylim(50,500)
    mixer.init()
    refs = []
    tests = []
    for f, r in zip(test_files, ref_files):
        print("analyzing "+f)

        ref = np.load(r)
        ref_f0 = np.array([item[1] for item in ref])
        ref_f0[ref_f0<60]=0
        f0 = analysis(f, 50, 400, 0.2)
        #try:
        f0 = f0[:len(ref_f0)]
        #except:
        ref_f0 = ref_f0[:len(f0)]
        print("vde:",vde(f0, ref_f0))
        plt.plot(f0, 'red')
        plt.plot(ref_f0, 'black')
        plt.show()
        refs.append(ref_f0)
        tests.append(f0)
    ref_f0= np.concatenate(refs, axis=None)
    test_f0= np.concatenate(tests, axis=None)

    print("vde:",vde(test_f0, ref_f0))
    print("gross:",gross(test_f0, ref_f0))
    print("mae:",mae(test_f0, ref_f0))