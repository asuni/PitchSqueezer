


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


def f0_cwt(f0_interp, plot=False):

    
    f0_interp = (f0_interp-np.mean(f0_interp)) #/ np.std(f0_interp)
    f0_padded = np.concatenate([f0_interp,f0_interp, f0_interp])
    f0_padded = np.pad(f0_interp, len(f0_interp), mode='reflect')
    wavelet = 'cmhat'
    nv = 2 # scales per octave
    n = 4096 # 12 octaves

    min_scale, max_scale = cwt_scalebounds(wavelet, N=n, preset='maximal')

    scales = make_scales(nv=nv, min_scale=min_scale, max_scale=max_scale, scaletype='log', wavelet=wavelet, N=n)
    scalogram, scales, *_ = cwt(f0_padded, wavelet=wavelet, fs=100, scales=scales)
    scalogram = np.real(scalogram)
    scalogram*=2.2/nv # empirical correction for reconstruction
    scalogram = scalogram[:,len(f0_interp):-len(f0_interp)]
    reduced = np.zeros((5, scalogram.shape[1]))
    reduced[0] = np.sum(scalogram[0*nv:3*nv], axis=0)
    reduced[1] = np.sum(scalogram[3*nv:5*nv], axis=0)
    reduced[2] = np.sum(scalogram[5*nv:7*nv], axis=0)
    reduced[3] = np.sum(scalogram[7*nv:9*nv], axis=0)
    reduced[4] = np.sum(scalogram[9*nv:], axis=0)

    if plot:
        plt.plot(f0_interp, label="original")
        plt.plot(np.sum(scalogram, axis=0), label="reconstructed")
        for i in range(0, 5):
            plt.plot(reduced[i]+i*0.5+0.5, color="black")
        plt.legend()
        plt.show()
    return reduced.T


def _hp_filter(sig, cutoff_frequency=60, order=4):
    from scipy.signal import butter, filtfilt
    b, a = butter(order, cutoff_frequency / (0.5 * 4000), btype='high', analog=False)
    # Apply the filter to the signal
    return filtfilt(b, a, sig)


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
    voiced[voiced==min_val] = np.nan

    if np.isnan(voiced[-1]):
        voiced[-1] = np.nanmin(voiced)
    if np.isnan(voiced[0]):
        voiced[0] = np.nanmean(voiced)
        
    # add helping points in long gaps
    nan_indices = np.isnan(voiced)
    st_i = np.where(~nan_indices[:-1] & nan_indices[1:])[0] + 1
    end_i = np.where(nan_indices[:-1] & ~nan_indices[1:])[0] + 1

  
    for i in range(0, len(end_i)):
    
        voiced[st_i[i]+1] = voiced[st_i[i]-1]-1 # constrains akima
        voiced[end_i[i]-1] = voiced[end_i[i]]-1 # constrains akima
        gap_len = end_i[i]-st_i[i]
        if gap_len > 40:
            gap_min = np.min([voiced[st_i[i]-1], voiced[end_i[i]]])
            voiced[end_i[i]-30] = gap_min - int(np.sqrt(gap_len))
       
    not_nan = np.logical_not(np.isnan(voiced))

    indices = np.arange(len(voiced))
    if method == 'pchip':
        interp = scipy.interpolate.pchip(indices[not_nan], voiced[not_nan])
    elif method =='akima':
        interp = scipy.interpolate.Akima1DInterpolator(indices[not_nan], voiced[not_nan])
    else:
        interp = scipy.interpolate.interp1d(indices[not_nan], voiced[not_nan], method)

    return interp(indices)


def _hz_to_semitones(hz_values, base_freq=50):
    return 12 * np.log2(hz_values / base_freq)


def _apply_target_rate(pitch_track, old_frame_rate, n_target_points):
    old_time_points = np.linspace(0, len(pitch_track) / old_frame_rate, num=len(pitch_track))
    new_time_points = np.linspace(0, old_time_points[-1], num=n_target_points)
    new_pitch_track = np.interp(new_time_points, old_time_points, pitch_track)
    return new_pitch_track

# if frame has energy at 1/2 or 1/3 of the max pitch candidate, move energy there
def _stack_f0(spec, min_hz, max_hz):
    max_ind= np.argmax(spec[:, min_hz:max_hz], axis=1)+min_hz 
    for i in range(0, spec.shape[0]):
        cand = max_ind[i]
        if cand > 3*min_hz and spec[i, int(cand/3.)] > spec[i, cand] * 0.1 and spec[i, int(cand/2.)] > spec[i, cand] *0.1 :
            spec[i, int(cand/3.)]+=spec[i, cand]*5.
        if cand > 2*min_hz and spec[i, int(cand/2.)] > spec[i, cand] * 0.1:
            spec[i, int(cand/2.)]+=spec[i, cand]*5.
    return spec

def _construct_window(mean_hz):
    #from scipy.signalget_window # import gaussian
    mean_hz=int(mean_hz)
    l_window = scipy.signal.windows.gaussian(mean_hz*2,mean_hz*0.6)
    r_window = scipy.signal.windows.gaussian(mean_hz*4, mean_hz*0.6)
    window = np.concatenate((l_window[:len(l_window)//2], r_window[len(r_window)//2:]))
    return window

def _add_slope(spec, min_hz=50, max_hz=500, steepness=1.):
    #increasing spectral slope, in order to make fundamental stand out more
    if min_hz<1:min_hz = 1
    spec[:, np.arange(min_hz, max_hz)] /=  np.arange(min_hz,max_hz) ** steepness
    return spec

def _get_max_track(spec, unvoiced_frames =[],min_hz=50, max_hz=500):
    f0 = np.argmax(spec[:, min_hz:max_hz], axis=1)+min_hz 
    f0[unvoiced_frames] = 0
    return f0


# main function
def track_pitch(utt_wav,min_hz=60, max_hz=500, voicing_thresh=0.3, target_rate=200, plot=False):
    """
    extract f0 track from speech wav file using synchrosqueezed spectrogram and frequency domain autocorrelation
    """
    
    # some internal constants, could be user params also
    SR = 4000.0          # sample rate should be high enough for spectral autocorrelation (3 harmonics form max_hz)
    INTERNAL_RATE = 100  # frames per second, 100 for speed, >=200 for accuracy
    SPEC_SLOPE = 1.25     # adjusting slope steeper will emphasize lower harmonics
    ACORR_WEIGHT = 3.    #
    VITERBI_PENALTY = 3  # larger values will provide smoother track but might cut through fast moving peaks
    MIN_VAL = 1.0e-20
    
    PLT_MAX_HZ = max_hz  
    orig_sr = librosa.get_samplerate(utt_wav)

    # get integer ratio between original and internal sample rate to avoid rounding problems
    if orig_sr in [11025, 22050, 44100]:
        SR = 4410

   
    # read wav file, downsample to 4000Hz, highpass filter to get rid of hum, and normalize
    sig, fs = librosa.load(utt_wav, sr=SR)
    sig = _hp_filter(sig, cutoff_frequency=80)
    sig = (sig-np.mean(sig)) / np.std(sig) 
    
    # do ffts on the signal
    frame_shift = int(round(SR/INTERNAL_RATE)) 
    ssq_fft ,fft2, *_ = ssq_stft(sig ,n_fft = int(SR), win_len=int(SR/4),hop_len=frame_shift)
 
    ssq_fft = abs(ssq_fft).T # personal preference for (time, hz) shape
   
     ### for voicing decision, use stft with shorter window                                                                                                                          
    short_win_fft, *_= ssq_stft(sig, n_fft=int(SR),win_len=int(SR/16),hop_len=frame_shift)
    short_win_fft = abs(short_win_fft).T
   
    if plot:
        fig, ax = plt.subplots(6, 1, sharex=True, sharey=True)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0.1, wspace = 0)
        energy2 = np.sum(short_win_fft[:,min_hz:max_hz], axis=1)
        unvoiced_frames = energy2 < voicing_thresh
        pic_fft = abs(fft2).T
      
        ax[0].imshow(np.log(pic_fft[:,:PLT_MAX_HZ]).T, aspect="auto", origin="lower")
        ax[0].plot(_get_max_track(pic_fft, unvoiced_frames), color="orange")
        ax[0].set_title("fft", loc="left", x=0.02, y=0.7, color="white")

    pic = scipy.ndimage.gaussian_filter(ssq_fft,[1,2])
    pic[pic<MIN_VAL] = MIN_VAL

    if plot:
        ax[1].imshow(np.log(pic[:,:PLT_MAX_HZ]).T, aspect="auto", origin="lower")
        ax[1].plot(_get_max_track(pic, unvoiced_frames), color="orange")    
        ax[1].set_title("+synchro-squeezing", loc="left",x=0.02, y=0.7, color="white")
    
    
    # frequency domain autocorrelation

    length = int(SR/2)
    acorr1 = librosa.autocorrelate(pic[:,:length], max_size=SR/2, axis=1)

    # bias toward lower peaks
    acorr1 *= np.linspace(2, 0, acorr1.shape[1])

    # add strongest autocorrelation frequency to spec, helps with weak or missing fundamental
    acorr_f0 = np.argmax(acorr1[:, min_hz:max_hz], axis=1)+min_hz
    acorr_f0 = scipy.signal.medfilt(acorr_f0, 3)
    for i in range(-3,3):
        pic[np.arange(pic.shape[0]), acorr_f0+i] += acorr1[np.arange(pic.shape[0]), acorr_f0+i]*ACORR_WEIGHT
        
    # multiply spec with the whole correlogram, mainly for denoising aperiodic sections
    pic[:, :length] *=acorr1
    
    if plot:
        ax[2].imshow(np.log(pic[:,:PLT_MAX_HZ]).T, aspect="auto", origin="lower")
        ax[2].plot(_get_max_track(pic, unvoiced_frames), color="orange")
        ax[2].set_title("+freq autocorrelation", loc="left",x=0.02, y=0.7, color="white")
   
    
    # voicing decision from autocorrelation and short window fft
    energy1 = np.sum(acorr1[:, min_hz:max_hz], axis=1)
    energy2 = np.sum(short_win_fft[:,min_hz:max_hz], axis=1)
    unvoiced_frames  = energy1 + energy2 < voicing_thresh
    # remove short sections
    unvoiced_frames = scipy.signal.medfilt(unvoiced_frames.astype('int'), 11).astype('bool')
    voiced_frames =  np.invert(unvoiced_frames)

    # dampen higher frequencies to reduce octave jumps up
    pic = _add_slope(pic, min_hz=0, max_hz=max_hz, steepness=SPEC_SLOPE)
      
    # reassign energy from h2 to h1
    pic = _stack_f0(pic, min_hz, max_hz)
    pic = _stack_f0(pic, min_hz, max_hz)

    if plot:
        ax[3].imshow(np.log(pic[:,:PLT_MAX_HZ]).T, aspect="auto", origin="lower")
        ax[3].plot(_get_max_track(pic, unvoiced_frames), color="orange")
        ax[3].set_title("+skew spectrum + harmonic reassignment", loc="left",x=0.02, y=0.7, color="white")


    # softy constrain the hypothesis space by windowing around initial median estimate
    mean_pitch = np.median(np.argmax(pic[voiced_frames, min_hz:max_hz], axis=1))+min_hz
    window = _construct_window(mean_pitch)
    pic[:, :len(window)]*=window

    if plot:
        ax[4].imshow(np.log(pic[:,:PLT_MAX_HZ]).T, aspect="auto", origin="lower")
        ax[4].plot(_get_max_track(pic, unvoiced_frames), color="orange")
        ax[4].set_title("+window around median pitch", loc="left",x=0.02, y=0.7, color="white")

    
    # narrow search space between observed min and max
    raw_f0 = np.argmax(pic[voiced_frames, min_hz:max_hz], axis=1)+min_hz 
    min_freq,max_freq = np.min(raw_f0), np.max(raw_f0)
    scales = np.arange(min_freq, max_freq, 1)
    scales = _hz_to_semitones(scales, min_freq)
    pic[pic<MIN_VAL] = MIN_VAL
    # viterbi search for the best path
    track = extract_ridges(pic[:,min_freq:max_freq].T,scales, penalty=VITERBI_PENALTY, n_ridges=1, transform="fft")  
    track = np.array(track).astype('float').flatten()+min_freq
    track[unvoiced_frames] = 0

    
    if plot:
        ax[5].imshow(np.log(pic[:,:PLT_MAX_HZ]).T, aspect="auto", origin="lower")
        ax[5].plot(track, color="orange", linestyle="dotted")
        ax[5].set_title("+viterbi", loc="left",x=0.02, y=0.7, color="white")
        plt.show()

    # fill unvoiced gaps and postprocess 
    interp_track = scipy.signal.medfilt(track, 9)
    interp_track = _interpolate_zeros(interp_track, method='akima')
   
    # combine interpolated unvoiced regions and unsmoothed voiced regions and smooth again
    interp_track[voiced_frames] = track[voiced_frames]
    interp_track = scipy.signal.medfilt(interp_track, 5)
    interp_track = _smooth(interp_track, 5)

    # convert to target frame rate 
    if target_rate != INTERNAL_RATE:
       
        # for compatibility with librosa pyin and pytorch stft
        n_target_frames = np.ceil(len(sig)*(orig_sr/SR) / round(orig_sr/target_rate)).astype('int')
        track = _apply_target_rate(track, INTERNAL_RATE, n_target_frames)
        interp_track = _apply_target_rate(interp_track, INTERNAL_RATE, n_target_frames)
        unvoiced_frames = _apply_target_rate(unvoiced_frames.astype('int'), INTERNAL_RATE, n_target_frames).astype('bool')
        track[unvoiced_frames] = 0 # if the iterpolation has smoothed track
   
    if plot:
        plt.plot(interp_track, linestyle="dotted", color="white", alpha=0.3)
        plt.plot(track, linestyle="dotted", color="white", label="squeezer")
        y, sr = librosa.load(utt_wav)
        f0_pyin, voiced_flag, voiced_probs = librosa.pyin(y,sr=sr, fmin=min_hz, fmax=max_hz, hop_length = round(sr/target_rate))
        plt.plot(f0_pyin, color="red", linestyle="dotted", label="pyin")
        if target_rate == INTERNAL_RATE:
            plt.imshow(np.log(pic[:,:PLT_MAX_HZ]).T, aspect="auto", origin="lower")
        print(len(f0_pyin), len(track))
        plt.legend()
        plt.show()

    return (track, interp_track)



def extract_to_file(utt_wav,min_hz=60, max_hz=500, voicing_thresh=0.3, target_rate=200, output_directory=None, wavelet=False, output_format="txt"):
    f0, if0 = track_pitch(utt_wav, min_hz, max_hz, voicing_thresh, target_rate)
    if wavelet:
         cwt_mat = f0_cwt(np.log(if0))
    if output_directory:
        out_name = output_directory+"/"+os.path.splitext(os.path.basename(utt_wav))[0]
    else:
        out_name = os.path.splitext(utt_wav)[0]
    if output_format == "txt":
        np.savetxt(out_name+".interp.txt", if0, fmt='%f')    
        np.savetxt(out_name+".f0.txt", f0, fmt='%f')
        if wavelet:
            np.savetxt(out_name+".cwt.txt", cwt_mat, fmt='%f')
            
    elif output_format == "pt":
        torch.save(torch.from_numpy(if0), out_name+".interp.pt")
        torch.save(torch.from_numpy(f0), out_name+".f0.pt")
        if wavelet:
            torch.save(torch.from_numpy(cwt_mat),out_name+".cwt.txt")
            

def main():

    import argparse, glob
    from joblib import Parallel, delayed
    from tqdm import tqdm
    parser = argparse.ArgumentParser(
        description="Command line tool for tracking pitch in speech files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add options
    parser.add_argument("-m", "--min_hz", default=60, type=int,
                        help="minimum f0 value of the data, around 50-80 for males, 100 -150 for females")
    parser.add_argument("-M", "--max_hz", default=500, type=int,
                        help="maximum f0 value of the data, around 200-300 for males, 300-500 for females")
    parser.add_argument("-t", "--voicing_thresh", default=0.2, type=float,
                        help="maximum f0 value of the data, around 200-300 for males, 300-500 for females")
    parser.add_argument("-r", "--frame_rate", default=100.0, type=float,
                        help="number of f0 values per second, (for many TTSs such as fastpitch 22050hz, this is 86.1326 (256 samples)")                 
    parser.add_argument("-j", "--nb_jobs", default=4, type=int,
                        help="Define the number of jobs to run in parallel")
    parser.add_argument("-w", "--wavelet", action="store_true",
                        help="Extract 5-scale continuous wavelet decomposition of f0")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="plot the stages of pitch of the algorithm")
    parser.add_argument("-f", "--output_format", default="txt", choices=["txt", "pt", "npy"], 
                        help="text file, pytorch tensor, numpy array" )
    parser.add_argument("-o", "--output_directory", default=None, type=str,
                        help="The output directory. If not specified, input directory will be used.")
    
    
    # Add arguments
    parser.add_argument("input", help="file or directory with audio files (.wav) to analyze")
    
    args = parser.parse_args()

     
    if str.lower(args.input).endswith(".wav"):
        input_files = [args.input]
    else:
        input_files = sorted(glob.glob(args.input + "/*.wav"))

    if args.plot:
        import random
        random.shuffle(input_files)
        for f in input_files:
            os.system("play -q "+f+ "&")
            f0, if0 = track_pitch(f,args.min_hz, args.max_hz, args.voicing_thresh, args.frame_rate, plot=True)
            if args.wavelet:
                f0_cwt(np.log(if0), plot=True)
            continue
        return

    if args.output_format == "pt":
        try:
            import torch
        except:
            print("to export torch tensors, you need to have torch installed. (pip install torch)")
            sys.exit(0)
    
    if args.output_directory:
        try:
            os.mkdir(args.output_directory)
        except:
            pass
    
    Parallel(n_jobs=args.nb_jobs)(delayed(extract_to_file)(f, 
            args.min_hz, 
            args.max_hz, 
            args.voicing_thresh,
            args.frame_rate,
            args.output_directory,
            args.wavelet,
            args.output_format) for f in tqdm(input_files))

if __name__ == "__main__":
    sys.exit(main())