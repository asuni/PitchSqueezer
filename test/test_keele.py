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

def get_errors(test_f0, ref_f0):
    
    voicing_errors = np.logical_xor(ref_f0 == 0, test_f0 == 0)
    voicing_errors = np.sum(voicing_errors)/len(ref_f0)*100
    voiced_frames = np.logical_and(ref_f0 > 0, test_f0 > 0)
    test_f0 = test_f0[voiced_frames]
    ref_f0 = ref_f0[voiced_frames]
    errors = np.abs(test_f0 - ref_f0)
    gross_errors = errors > 0.2 * ref_f0
    percentage_gross_errors = (np.sum(gross_errors) / len(errors)) * 100
    fine_errors = errors[~gross_errors]
    mean_fine_errors = np.mean(fine_errors)
    return voicing_errors, percentage_gross_errors, mean_fine_errors


def analysis(wav, fmin, fmax, method = "squeezer"):
    
    if method == "squeezer":
        f0_ps, if0_ps = ps.track_pitch(wav, min_hz=fmin, max_hz=fmax, voicing_thresh=0.4,frame_rate=100)
        return f0_ps
    
    elif method == "yaapt":
        signal = basic.SignalObj(wav)
        f0_pyaapt = pYAAPT.yaapt(signal, **{'f0_min' : fmin, 'f0_max':fmax, 'frame_space' : 10.0})
        f0_pyaapt = f0_pyaapt.samp_values
        f0_pyaapt = np.pad(f0_pyaapt, 1, 'edge')
        return f0_pyaapt
   
    elif method == "pyin":
        x, fs = librosa.load(wav, sr=None)
       
        f0_pyin, voiced_flag, voiced_probs = librosa.pyin(x,sr=fs, fmin=fmin, fmax=fmax, frame_length=int(fs/16), hop_length = int(fs/100))
        f0_pyin[np.isnan(f0_pyin)]=0
        return f0_pyin

    elif method == "praat":
        sound = parselmouth.Sound(wav)
        pitch = sound.to_pitch_ac(time_step = 0.01, pitch_floor=fmin, pitch_ceiling=fmax)
        pitch = pitch.selected_array['frequency']
        pitch = np.pad(pitch, 2, 'edge')
        return pitch
    
    else:
        print(method+" not implemented")
        return None
    
  
    
if __name__ == "__main__":
    import sys, glob, time
    test_files = sorted(glob.glob(sys.argv[1]+"/*/signal.wav"))
    ref_files = sorted(glob.glob(sys.argv[1]+"/*/*.npy"))

  
    for method in ("praat", "squeezer", "yaapt", "pyin"):
    
        print("evaluating "+method,end=": ", flush=True)
        start = time.time()
        refs = []
        tests = []
        for f, r in zip(test_files, ref_files):
            print(".", end="",flush=True)
            #print("analyzing "+f)

            ref = np.load(r)
            ref_f0 = np.array([item[1] for item in ref])
            ref_f0[ref_f0<60]=0
            f0 = analysis(f, 50, 500, method)
            # match lengths (for all methods, 1 frame off only)
            f0 = f0[:len(ref_f0)]
            ref_f0 = ref_f0[:len(f0)]
            
            refs.append(ref_f0)
            tests.append(f0)

        ref_f0= np.concatenate(refs, axis=None)
        test_f0= np.concatenate(tests, axis=None)

        vde, gpe, fpe = get_errors(test_f0, ref_f0)
        print(" done in ", time.time()-start, "seconds")
        print("vde " +str(vde)+"% gpe "+str(gpe)+"% fpe "+str(fpe)+"hz\n")
  
   