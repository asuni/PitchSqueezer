import pitch_squeezer as ps
import librosa
import numpy as np
import time
import matplotlib.pyplot as plt
import pyworld as pw
import soundfile as sf
import os
from pygame import mixer

def _highlight(f0s, index):
    colors = ["green", "blue", "red"]
    for i in range(len(f0s)):
        f0s[i][(f0s[i]==0)] = np.nan
        if i == index:
            plt.plot(f0s[i], color=colors[i], linewidth=2)
        else:
            plt.plot(f0s[i], color=colors[i],alpha=0.15)
        plt.pause(0.1)

def play(wav):
    
    def play_wav(wav):
        #os.system("play "+wav)
        #print(wav)
        mixer.music.load(wav)
        mixer.music.play()
   
    from threading import Thread
    # Play Music on Separate Thread (in background)
    #music_thread = Thread(target=play_wav, args=([wav]))
    #music_thread.start()
    play_wav(wav)
    input("press ENTER to stop playback")
    mixer.music.stop()
    #music_thread.join()
   

def anasyn(wav):
    

    start_time = time.time()
    f0_ps, if0_ps = ps.track_pitch(wav, voicing_thresh=0.25, target_rate=200)
    print("squeezer analysis done in ", time.time()-start_time, "seconds")

    start_time = time.time()
    x, fs = librosa.load(wav, sr=None)
    f0_pyin, voiced_flag, voiced_probs = librosa.pyin(x,sr=fs, fmin=50, fmax=500, hop_length = int(fs/200))
    print("pyin analysis done in ", time.time()-start_time, "seconds")
    

    start_time = time.time()
    x = np.array(x, dtype=np.double)
    _f0, t = pw.dio(x, fs)    # raw pitch extractor
    f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
    print("world f0 analysis done in ", time.time()-start_time, "seconds")
    plt.ion()
    #print(f0_pyin.shape, f0_ps.shape)
    f0_pyin = f0_pyin[:len(f0)]
    f0_ps = f0_ps[:len(f0)]
    
    sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
    print(sp.shape)
    sp[f0_ps>0] = np.mean(sp[f0_ps>0], axis=0)
    ap = pw.d4c(x, f0, t, fs)         # extract aperiodicity
    ap[:] = 0.
   
    f0_pyin[np.isnan(f0_pyin)] = 0
   
    y1 = pw.synthesize(f0_pyin, sp, ap, fs)
    y2 = pw.synthesize(f0_ps, sp, ap, fs) # synthesize an utterance using the parameters
    y3 = pw.synthesize(f0, sp, ap, fs) # synthesize an utterance using the parameters
    #ap = ap[:len(f0_ps)]
    #sp = sp[:len(f0_ps)]
    
    y2 = pw.synthesize(f0_ps, sp, ap, fs) # synthesize an utterance using the parameters
    sf.write("output.wav", y1, fs)
    sf.write("output2.wav", y2, fs)
    sf.write("output3.wav", y3, fs)
    #librosa.output.write_wav("output.wav", generated_audio, 4000)
    f0s = (f0_pyin, f0_ps, f0)

    print("synthesized with pyin f0...")
    plt.cla()
    plt.title("pyin")
    _highlight(f0s, 0)
    play("output.wav")

    print("synthesized with pitchsqueezer f0...")
    plt.cla()
    plt.title("pitchsqueezer")
    _highlight(f0s, 1)
    play("output2.wav")
  

    print("synthesized with World f0...")
    plt.cla()
    plt.title("world")
    _highlight(f0s, 2)
  
    play("output3.wav")
    plt.cla()
    
if __name__ == "__main__":
    import sys, glob
    test_files = sorted(glob.glob(sys.argv[1]+"/*.wav"))
    import random
    random.shuffle(test_files)
    plt.figure(figsize=(12,4))
    mixer.init()
    for f in test_files:
        play(f)
        print("analyzing "+f)
       
        anasyn(f )
    sys.exit(main())
