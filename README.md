# PitchSqueezer
A pitch tracker for speech, using synchro-squeezed stft and frequency domain autocorrelation, designed to analyze imperfect recordings of spontaneous speech.
Relies heavily on the nice [ssqueezepy](https://github.com/OverLordGoldDragon/ssqueezepy/tree/master/ssqueezepy]) package.


Features:
* (mostly) better quality than popular Python trackers (pyin, pyaapt) 
* robust to creaky voices and bad recordings
* does not need finetuning for min and max f0, works fine with both low and high pitched voices
* acceptable speed (~ 10x faster than librosa.pyin)
* (alternatively) a continuous pitch track, filling unvoiced gaps relatively naturally
* a wavelet decomposition of the pitch track (reversible, except for mean value) 
* compatible with librosa's pyin and pytorch_audio spectrograms, regarding number of frames
* a command line tool for parallel batch processing of directories (as well as API)

Installation: 
```
pip install pitchsqueezer
```

Examples of basic usage:
```
;; Command line, extract f0 for all wavs in a directory using 10 ms frame shift, save as numpy files
$ pitchsqueezer path/to/wavs/ -r 100 -f npy

;; API
import pitch_squeezer as ps
f0, if0 = ps.track_pitch(input_file, min_hz=50, max_hz=500)
f0_cwt = ps.f0_cwt(if0)
```
docs: https://asuni.github.io/PitchSqueezer/

Visualization of the method: <img src="images/Figure_1.png">



 comparison with librosa.pyin on creaky female voice: <img src="images/squeezer_vs_pyin.png">
