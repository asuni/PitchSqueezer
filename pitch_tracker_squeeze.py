


from ssqueezepy import ssq_cwt, cwt, ssq_stft, stft, Wavelet, extract_ridges
from ssqueezepy.utils import cwt_scalebounds, make_scales

import sys, os
#os.environ['SSQ_GPU'] = '1'
import numpy as np
from scipy.io import wavfile
import scipy.signal
import scipy.interpolate
import librosa
import matplotlib.pyplot as plt


def _f0_cwt(f0_interp):

    
    f0_interp = (f0_interp-np.mean(f0_interp)) #/ np.std(f0_interp)
    f0_padded = np.concatenate([f0_interp,f0_interp, f0_interp])
    f0_padded = np.pad(f0_interp, len(f0_interp), mode='reflect')
    wavelet = 'cmhat'
    nv = 2 # scales per octave
    n = 4096 # 12 octaves

    min_scale, max_scale = cwt_scalebounds(wavelet, N=n, preset='maximal')

    scales = make_scales(nv=nv, min_scale=min_scale, max_scale=max_scale, scaletype='log', wavelet=wavelet, N=n)
    scalogram, scales, *_ = cwt(f0_padded, wavelet=wavelet, fs=100, scales=scales)

    scalogram*=2.2/nv # empirical correction for reconstruction
    scalogram = scalogram[:,len(f0_interp):-len(f0_interp)]
    reduced = np.zeros((5, scalogram.shape[1]))
    reduced[0] = np.sum(scalogram[0*nv:3*nv], axis=0)
    reduced[1] = np.sum(scalogram[3*nv:5*nv], axis=0)
    reduced[2] = np.sum(scalogram[5*nv:7*nv], axis=0)
    reduced[3] = np.sum(scalogram[7*nv:9*nv], axis=0)
    reduced[4] = np.sum(scalogram[9*nv:], axis=0)

    if DEBUG:
        for i in range(0, 5):
            plt.plot(reduced[i]+i, color="black")
    
        plt.figure()
        plt.plot(np.sum(scalogram, axis=0), label="reco")
        plt.plot(np.sum(scalogram[2:], axis=0), label="reco_sm")

        plt.plot(f0_interp, label="orig")
        plt.legend()
        plt.show()
    return reduced

def _cepstrum(spec, sr = 4000, minF0 = 50, maxF0 = 500):
    from scipy.fftpack import irfft, rfft, ifft, fft
    ceps =  abs(irfft(spec))
 
    ceps *= np.arange(ceps.shape[1]) ** -0.025
    max_i = int(sr / minF0)
    min_i = int(sr / maxF0)

    max_indices = np.argmax(ceps[:, min_i:max_i], axis=1)+min_i
    max_indices = scipy.signal.medfilt(max_indices, 3)
    mag = ceps[np.arange(ceps.shape[0]), max_indices]
  
    f0 = sr/max_indices
   
    return f0.astype(int), mag


        
def _smooth(params, win, type="HAMMING"):
    
    """
    gaussian type smoothing, convolution with hamming window
    """
    win = int(win+0.5)
    if win >= len(params)-1:
        win = len(params)-1
    if win % 2 == 0:
        win+=1
    s = np.r_[params[win-1:0:-1],params,params[-1:-win:-1]]
    w = np.hamming(win)
    y = np.convolve(w/w.sum(),s,mode='valid')
    return y[int(win/2):-int(win/2)]

    
def _interpolate_zeros(params, method='pchip', min_val = 0):


    voiced = np.array(params, float)
    for i in range(0, len(voiced)):
        if voiced[i] == min_val:
            voiced[i] = np.nan

    if np.isnan(voiced[-1]):
        voiced[-1] = np.nanmin(voiced)
    if np.isnan(voiced[0]):
        voiced[0] = np.nanmean(voiced)

    not_nan = np.logical_not(np.isnan(voiced))

    indices = np.arange(len(voiced))
    if method == 'spline':
        interp = scipy.interpolate.UnivariateSpline(indices[not_nan],voiced[not_nan],k=2,s=0)
        # return voiced parts intact
        smoothed = interp(indices)
        for i in range(0, len(smoothed)):
            if not np.isnan(voiced[i]) :
                smoothed[i] = params[i]
        return smoothed
    elif method =='pchip':
        interp = scipy.interpolate.pchip(indices[not_nan], voiced[not_nan])
    else:
        interp = scipy.interpolate.interp1d(indices[not_nan], voiced[not_nan], method)
    return interp(indices)




def _hz_to_semitones(hz_values, base_freq=50):
    return 12 * np.log2(hz_values / base_freq)

def _apply_target_rate(track, internal_rate, target_rate):

    existing_time_intervals = np.arange(0, len(track) * 5, 1./internal_rate*1000)
    new_time_intervals = np.arange(0, len(track) * 5, 1./target_rate*1000)
    new_track = np.interp(new_time_intervals, existing_time_intervals, track)

    return new_track

# if frame has energy at 1/2 or 1/3 of the max pitch candidate, move energy there
def _stack_f0(spec, min_hz, max_hz):
    
    
    max_ind= np.argmax(spec[:, min_hz:max_hz], axis=1)+min_hz 
    for i in range(0, spec.shape[0]):
        cand = max_ind[i]

        if cand > 2*min_hz and spec[i, int(cand/2.)] > spec[i, cand] * 0.1:
            spec[i, int(cand/2.)]+=spec[i, cand]*5.
            spec[i, cand-6:cand+6] = 0 
    
    spec =  scipy.ndimage.gaussian_filter(spec,[1,1])
    return spec


def _construct_window(mean_hz):
    from scipy.signal import gaussian
    mean_hz=int(mean_hz)
    l_window = gaussian(mean_hz*2, std=mean_hz*0.2)
    r_window = gaussian(mean_hz*4, std=mean_hz*0.75)
    window = np.concatenate((l_window[:len(l_window)//2], r_window[len(r_window)//2:]))

    return window

def _add_slope(spec, min_hz=50, max_hz=500, steepness=1.):
    #increasing spectral slope, in order to make fundamental stand out more
    
    spec[:, np.arange(min_hz, max_hz)] /=  np.arange(min_hz,max_hz) ** steepness
    
    return spec


def inst_freq_pitch(utt_wav,min_hz=60, max_hz=500, voicing_thresh=0.3, target_rate=200, cwt=False , DEBUG=False):
    """
    extract f0 track from speech wav file using synchrosqueezed spectrogram and frequency domain autocorrelation
    """
    print(os.path.splitext(os.path.basename(utt_wav))[0])

    internal_rate = 200

    
    # some internal constants, could be user params also
    SPEC_SLOPE = 1.5
    ACORR_WEIGHT = 3.
    VITERBI_PENALTY = 4
    
    # read wav file, downsample to 4000Hz and normalize

    sr = 4000.0
    params, fs = librosa.load(utt_wav, sr=sr)
    params = (params-np.mean(params)) / np.std(params) 
    DEC = int(round(sr/internal_rate)) # decimation factor


    #print("fft...")

    ssq_fft ,fft2, *_ = ssq_stft(params, n_fft=int(sr),win_len=int(sr/4),hop_len=DEC)
    ssq_fft = abs(ssq_fft).T # personal preference for (time, hz) shape
    pic = scipy.ndimage.gaussian_filter(ssq_fft,[1,3])

    
    ### for voicing decision, use stft with shorter window
    short_win_fft, *_= ssq_stft(params, n_fft=int(sr),win_len=int(sr/16),hop_len=DEC)
    short_win_fft = abs(short_win_fft).T

    
    #print("frequency domain autocorrelation...")
    length = int(sr/2)
    length = int(np.min([max_hz*3, 2000]))
    acorr1 = librosa.autocorrelate(pic[:,:length], max_size=sr/4, axis=1)
    # bias toward lower peaks
    acorr1*=  np.linspace(1, -1., acorr1.shape[1])

   
    # voicing decision from autocorrelation and short window fft
    energy1 = np.sum(acorr1[:, min_hz:max_hz], axis=1)
    energy2 = np.sum(short_win_fft[:,min_hz:max_hz], axis=1)
    unvoiced_frames  = energy1 + energy2 < voicing_thresh
    voiced_frames =  energy1 + energy2 >= voicing_thresh

    # add strongest autocorrelation frequency, helps with weak or missing fundamental
    acorr_f0= np.argmax(acorr1[:, min_hz:max_hz], axis=1)+min_hz
    acorr_f0 = scipy.signal.medfilt(acorr_f0, 3)

    for i in range(-3,3):
        pic[np.arange(pic.shape[0]), acorr_f0+i] += acorr1[np.arange(pic.shape[0]), acorr_f0+i]*ACORR_WEIGHT

    
    # multiply spec with the whole correlogram, mainly for denoising aperiodic sections
    acorr1 = scipy.ndimage.gaussian_filter(acorr1,[5,3])
    pic[:, :int(sr/4)] *=acorr1
          
            
    # reassign energy from h2 and h3 to f0
    pic = _stack_f0(pic, min_hz, max_hz)
    #pic = _stack_f0(pic, min_hz, max_hz)
    
    # dampen higher frequencies to reduce octave jumps up
    pic = _add_slope(pic, min_hz=min_hz, max_hz=max_hz, steepness=SPEC_SLOPE)
    raw_f0 = np.argmax(pic[voiced_frames, min_hz:max_hz], axis=1)+min_hz
    mean_pitch = np.median(raw_f0)

    raw_f0 = np.argmax(pic[:, min_hz:max_hz], axis=1)+min_hz

    plt.figure()
    raw_f0[unvoiced_frames] = 0
    plt.plot(raw_f0)



    # softy constrain the hypothesis space by windowing around initial median estimate
    window = _construct_window(mean_pitch)   
    pic[:, :len(window)]*=window

    raw_f0 = np.argmax(pic[:, min_hz:max_hz], axis=1)+min_hz

    raw_f0[unvoiced_frames] = 0
    plt.plot(raw_f0, label="after window")
    plt.legend()
    plt.show()

    #print("tracking")
    # narrow search space
    raw_f0 = np.argmax(pic[voiced_frames, min_hz:max_hz], axis=1)+min_hz

    min_freq = np.min(raw_f0)
    max_freq = np.max(raw_f0)
    scales = np.arange(min_freq, max_freq, 1)
    scales = _hz_to_semitones(scales, min_freq)
    # viterbi search for the best path

    track = extract_ridges(pic[:,min_freq:max_freq].T,scales, penalty=VITERBI_PENALTY, n_ridges=1, transform="fft")  
    track = np.array(track).astype('float').flatten()+min_freq #.flatten()+min_hz


    # fill unvoiced gaps and postprocess
    track[unvoiced_frames] = 0
    raw_f0 = np.array(track)

    # process a bit to get smoother interpolation 
    uv_track = scipy.signal.medfilt(track,9)
    uv_track = _interpolate_zeros(uv_track, method='pchip')
    uv_track = _smooth(uv_track,5)
    
    # combine interpolated unvoiced regions and unsmoothed voiced regions and smooth again
    #track = scipy.ndimage.generic_filter(track, np.nanmedian, 3) #medfilt(track,3)
    track[unvoiced_frames] = uv_track[unvoiced_frames]
    track = scipy.signal.medfilt(track,5)   
    track = _smooth(track,3)

    # convert to target frame rate 
    if target_rate != internal_rate:
        track = _apply_target_rate(track, internal_rate, target_rate)
        unvoiced_frames = _apply_target_rate(unvoiced_frames, internal_rate, target_rate).astype('bool')
        track = scipy.signal.medfilt(track,1)
   
    interpolated_track = np.array(track)
    if cwt:
        cwt = _f0_cwt(np.log(interpolated_track))

    if DEBUG or 1==1:
        plt.plot(interpolated_track)

        plt.plot(raw_f0)
        pic[:,:min_hz] = 0
        plt.imshow(np.log(pic[:, :max_hz]).T, aspect="auto", origin="lower")
        plt.show()
    if cwt:
        return (interpolated_track, unvoiced_frames, cwt)
    return (interpolated_track,unvoiced_frames)


def extract_to_file(utt_wav,min_hz=60, max_hz=500, voicing_thresh=0.3, target_rate=200, output_directory=None, output_format="txt"):
    f0_track, unvoiced_frames = inst_freq_pitch(utt_wav, min_hz, max_hz, voicing_thresh, target_rate)
    f0_track_voiced = np.array(f0_track)
    f0_track_voiced[unvoiced_frames] = 0
    
    
    if output_directory:
        out_name = output_directory+"/"+os.path.splitext(os.path.basename(utt_wav))[0]
    else:
        out_name = os.path.splitext(utt_wav)[0]
        print("this output", out_name)
    if output_format == "txt":
        np.savetxt(out_name+".interp.txt", f0_track, fmt='%f')    
        np.savetxt(out_name+".f0.txt", f0_track_voiced, fmt='%f')
    elif output_format == "pt":
        torch.save(torch.from_numpy(f0_track), out_name+".interp.pt")
        torch.save(torch.from_numpy(f0_track_voiced), out_name+".f0.pt")


if __name__ == "__main__":
    import argparse, glob
    from joblib import Parallel, delayed
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description="Command line application to analyze prosody using wavelets.")

    # Add options
    parser.add_argument("-m", "--min_hz", default=60, type=int,
                        help="minimum f0 value of the data, around 50-80 for males, 100 -150 for females")
    parser.add_argument("-M", "--max_hz", default=500, type=int,
                        help="maximum f0 value of the data, around 200-300 for males, 300-500 for females")
    parser.add_argument("-t", "--voicing_thresh", default=0.3, type=float,
                        help="maximum f0 value of the data, around 200-300 for males, 300-500 for females")
    parser.add_argument("-r", "--frame_rate", default=86.1326, type=float,
                        help="number of f0 values per second, (for fastpitch 22050hz, this is 86.1326 (256 samples)")                 
    parser.add_argument("-j", "--nb_jobs", default=8, type=int,
                        help="Define the number of jobs to run in parallel")
    
    parser.add_argument("-f", "--output_format", default="txt", choices=["txt", "pt", "npy"])
    parser.add_argument("-o", "--output_directory", default=None, type=str,
                        help="The output directory. If not specified, the tool will output the result in a .f0 and .f0.interp file in the same directory as the wave files")
    
    
    # Add arguments
    parser.add_argument("input", help="directory with wave files")
    
    args = parser.parse_args()
    if args.output_format == "pt":
        import torch
    
    if args.output_directory:
        try:
            os.mkdir(args.output_directory)
        except:
            pass
            
    input_files = sorted(glob.glob(args.input + "/*.wav"))
    
    # test
    #"""
    for f in input_files:
        inst_freq_pitch(f,args.min_hz, args.max_hz, args.voicing_thresh, args.frame_rate, DEBUG=True)
        
    #"""
    Parallel(n_jobs=args.nb_jobs)(delayed(extract_to_file)(f, 
            args.min_hz, 
            args.max_hz, 
            args.voicing_thresh,
            args.frame_rate,
            args.output_directory, 
            args.output_format) for f in tqdm(input_files))
