


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
    """
    Decomposes continuous f0 track to 5 scales using continuous wavelet transform, with mexican hat mother wavelet

    Args:
        f0_interp (np.array):  an array containing log(f0) values
        plot (bool):    visualize the wavelet transform
       
    Returns:
        - scalogram (np.array with shape(len(f0_interp),5)): five scales corresponding roughly to phone, 
        syllable, word, phrase and utterance level movement of pitch. 
        np.sum(scalogram, axis=0) produces the original pitch track with mean value subtracted.

 
    """
    
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
        fig, ax = plt.subplots(6, 1, sharex=True, sharey=True)
        plt.subplots_adjust(top = 1., bottom = 0, right = 1, left = 0, hspace = 0., wspace = 0)
        scales = ["phone", "syllable", "word", "phrase", "utterance"]
        for i in range(0, 5):
            ax[i].plot(reduced[4-i], color="gray")
            ax[i].set_title(scales[4-i], loc="left", y=0.5) #, x=0.02, y=0.1, size=0.5)
           
        ax[5].plot(f0_interp, label="original", color="black")
        ax[5].plot(np.sum(reduced[1:], axis=0), color="red", label="reconstructed")
        plt.legend()
        plt.show()
    return reduced.T



def _hp_filter(sig, cutoff_frequency=60, order=2):
    from scipy.signal import butter, filtfilt
    b, a = butter(order, cutoff_frequency / (0.5 * 4000), btype='high', analog=False)
    # Apply the filter to the signal
    return filtfilt(b, a, sig)

def _smooth(params, win):
    
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

def _construct_window(mean_hz, bins_per_hz):
    #from scipy.signalget_window # import gaussian
    mean_hz=int(mean_hz)
    l_window = scipy.signal.windows.gaussian(mean_hz*2,mean_hz*0.6*bins_per_hz)
    r_window = scipy.signal.windows.gaussian(mean_hz*4, mean_hz*2.*bins_per_hz)
    window = np.concatenate((l_window[:len(l_window)//2], r_window[len(r_window)//2:]))
    return window

def _add_slope(spec, min_hz=50, max_hz=500, steepness=1.):
    #increasing spectral slope, in order to make fundamental stand out more
    if min_hz<1:min_hz = 1
    spec[:, np.arange(min_hz, max_hz)] /=  np.arange(min_hz,max_hz) ** steepness
    return spec

def _get_max_track(spec, unvoiced_frames =[],min_hz=50, max_hz=500):
    f0 = np.argmax(spec[:, int(min_hz):int(max_hz)], axis=1)+min_hz 
    f0[unvoiced_frames] = 0
    return f0

def _remove_outliers(track):
    fixed = np.array(track)
    mean_track = scipy.signal.medfilt(track, 31)
    try:
        mean_track = _interpolate_zeros(mean_track, 'linear')
    except:
        return fixed
    mean_track = _smooth(mean_track, 600)
   
    fixed[fixed<mean_track*0.8]=0 
    if 1==2 and len(fixed[fixed==0])!=len(track[track==0]):
        plt.figure()
        plt.plot(mean_track)
        plt.plot(track)
        plt.plot(fixed)
        plt.show()
    return fixed

def _plt(spec, uv_frames, min=0, max=500, ax=None, title=""):
    ax.imshow(np.log(spec[:,int(min):int(max)]).T, aspect="auto", origin="lower")
    ax.plot(_get_max_track(spec, uv_frames, max_hz=max), color="orange")
    ax.set_title(title, loc="left", x=0.02, y=0.7, color="white")




# main function
def track_pitch(utt_wav ,min_hz=60,max_hz=500, voicing_thresh=0.1,target_rate=100,plot=False):
    """
    Extracts f0 track from speech .wav file using synchrosqueezed spectrogram and frequency domain autocorrelation.
    Args:
        utt_wav (str): path to the audio file
        min_hz (int): minimum f0 value to consider
        max_hz (int): maximum f0 value to consider
        voicing_thresh (float): voicing decision tuner, 0.:all voiced, 1.:nothing voiced
        target_rate (float): number of pitch values per second to output
        plot (bool): visualize the analysis process

    Returns:
        tuple containing 2 arrays with length  ceil(len(wav) / floor(sr/target_rate))
        - track (np.array): array containing f0 values, unvoiced frames=0. 
        - interp_track (np.array)): array containing f0 values with unvoiced gaps filled using interpolation

    Example:
        ```
        import pitch_squeezer as ps
        f0, if0 = ps.track_pitch("test.wav", target_rate=200)
        lf0_cwt = ps.f0_cwt(np.log(if0))
        ```
    """
    
    # some internal constants, could be user params also
    SR = 4000.0          # sample rate should be high enough for spectral autocorrelation (3 harmonics form max_hz)
    INTERNAL_RATE = 100  # frames per second, 100 for speed, >=200 for accuracy
    BINS_PER_HZ = 1.     # determines the frequency resolution of the generated track, slows down rapidly if increased > 2
    SPEC_SLOPE = 1.25   # adjusting slope steeper will emphasize lower harmonics
    ACORR_WEIGHT = 3.    #
    VITERBI_PENALTY = 3/BINS_PER_HZ  # larger values will provide smoother track but might cut through fast moving peaks
    MIN_VAL = 1.0e-20
    min_hz_bin = int(min_hz * BINS_PER_HZ)
    max_hz_bin = int(max_hz * BINS_PER_HZ)
    orig_sr = librosa.get_samplerate(utt_wav)

    # get integer ratio between original and internal sample rate to avoid rounding problems
    # actually no, then frame_shift will cause rounding issues with 100 frames / s
    # if orig_sr in [11025, 22050, 44100]:
    #    SR = 4410


    # read wav file, downsample to 4000Hz, highpass filter to get rid of hum, and normalize
    sig, orig_sr = librosa.load(utt_wav, sr=None)
    orig_sig_len = len(sig) # needed for target frame rate conversion
    sig = librosa.resample(sig, orig_sr=orig_sr, target_sr=SR)
    sig = _hp_filter(sig, cutoff_frequency=70)
    sig = (sig-np.mean(sig)) / np.std(sig) 
    
    # do ffts on the signal
    frame_shift = round(SR/INTERNAL_RATE)
    ssq_fft ,fft2, *_ = ssq_stft(sig ,n_fft = int(SR*BINS_PER_HZ), win_len=int(SR/4),hop_len=frame_shift)

    ssq_fft = abs(ssq_fft).T # personal preference for (time, hz) shape

   
    if plot:
        fig, ax = plt.subplots(6, 1, sharex=True, sharey=True)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0.1, wspace = 0)
        pic_fft = abs(fft2).T
        energy2 = np.sum(pic_fft[:,min_hz:max_hz], axis=1)
        unvoiced_frames = energy2 < voicing_thresh*1000
    
        pic_fft = abs(fft2).T
        _plt(pic_fft, unvoiced_frames, max=max_hz_bin, ax=ax[0], title="fft")
      
    pic = scipy.ndimage.gaussian_filter(ssq_fft,[1,1*BINS_PER_HZ])
    pic[pic<MIN_VAL] = MIN_VAL

    if plot:
         _plt(pic, unvoiced_frames, max = max_hz_bin, ax=ax[1], title="synchro-squeezing")
      
    # frequency domain autocorrelation
    length = int(SR*BINS_PER_HZ/2)
    acorr1 = librosa.autocorrelate(pic[:,:length], max_size=int(SR/2*BINS_PER_HZ), axis=1)

    # bias toward lower peaks
    acorr1 *= np.linspace(2, 0, acorr1.shape[1])

    # add strongest autocorrelation frequency to spec, helps with weak or missing fundamental
    acorr_f0 = np.argmax(acorr1[:, min_hz_bin:max_hz_bin], axis=1)+min_hz_bin
    acorr_f0 = scipy.signal.medfilt(acorr_f0, 3)
    for i in range(-3,3):
        pic[np.arange(pic.shape[0]), acorr_f0+i] += acorr1[np.arange(pic.shape[0]), acorr_f0+i]*ACORR_WEIGHT
    
    # multiply spec with the whole correlogram, mainly for denoising aperiodic sections
    pic[:, :length] *=acorr1
    
    if plot:
        _plt(pic, unvoiced_frames, max = max_hz_bin, ax=ax[2], title="+freq autocorrelation")
    
    # voicing decision from autocorrelation and short window fft
    acorr_energy = np.sum(acorr1[:, min_hz_bin:max_hz_bin*2], axis=1)
   
    voicing_strength = np.log(acorr_energy+1.)
    unvoiced_frames  = voicing_strength < voicing_thresh

    # remove short sections
    unvoiced_frames = scipy.signal.medfilt(unvoiced_frames.astype('int'), 11).astype('bool')
    voiced_frames =  np.invert(unvoiced_frames)

    # dampen higher frequencies to reduce octave jumps up
    pic = _add_slope(pic, min_hz=0, max_hz=max_hz_bin, steepness=SPEC_SLOPE)
      
    # reassign energy from h2 to h1
    pic = _stack_f0(pic, min_hz_bin, max_hz_bin)
    pic = _stack_f0(pic, min_hz_bin, max_hz_bin)

    if plot:
        _plt(pic, unvoiced_frames, max = max_hz_bin, ax=ax[3], title="+skew spectrum + harmonic reassignment")

    # softy constrain the hypothesis space by windowing around initial median estimate
    mean_pitch = np.median(np.argmax(pic[voiced_frames, min_hz_bin:max_hz_bin], axis=1))+min_hz_bin
    window = _construct_window(mean_pitch, BINS_PER_HZ)
    pic[:, :len(window)]*=window

    if plot:
         _plt(pic, unvoiced_frames, max = max_hz_bin, ax=ax[4], title="+window around median pitch")
       

    # narrow search space between observed min and max
    raw_f0 = np.argmax(pic[voiced_frames, min_hz_bin:max_hz_bin], axis=1)+min_hz_bin
    min_freq,max_freq = np.min(raw_f0), np.max(raw_f0)
    scales = np.arange(min_freq, max_freq, 1)
    scales = _hz_to_semitones(scales, min_freq)
    pic[pic<MIN_VAL] = MIN_VAL
   
    # viterbi search for the best path
    track = extract_ridges(pic[:,min_freq:max_freq].T,scales, penalty=VITERBI_PENALTY, n_ridges=1, transform="fft")  
    track = np.array(track).astype('float').flatten()+min_freq

    track[unvoiced_frames] = 0
    if np.all(track == 0):
        print(utt_file+" all unvoiced!.")
        return (track, track)
    if plot:
        ax[5].imshow(np.log(pic[:,:max_hz_bin]).T, aspect="auto", origin="lower")
        ax[5].plot(track, color="orange", linestyle="dotted")
        ax[5].set_title("+viterbi", loc="left",x=0.02, y=0.7, color="white")
        plt.show()


    # fill unvoiced gaps and postprocess 

    interp_track = _remove_outliers(track)
    
    _interpolate_zeros(interp_track, method='akima')
    interp_track = _interpolate_zeros(interp_track, method='akima')
    interp_track = scipy.signal.medfilt(interp_track, 5)
    interp_track = _smooth(interp_track, 3)
 
    # convert to target frame rate 
    #if target_rate != INTERNAL_RATE:
       
        # for compatibility with librosa pyin and pytorch stft
    
    n_target_frames = np.ceil(orig_sig_len/np.floor(orig_sr/target_rate)).astype('int')
    track = _apply_target_rate(track, INTERNAL_RATE, n_target_frames)
    interp_track = _apply_target_rate(interp_track, INTERNAL_RATE, n_target_frames)
    unvoiced_frames = _apply_target_rate(unvoiced_frames.astype('int'), INTERNAL_RATE, n_target_frames).astype('bool')
    track[unvoiced_frames] = 0 # if the iterpolation has smoothed track

    if plot:
        #if target_rate == INTERNAL_RATE:
        #    plt.imshow(np.log(pic[:,:max_hz_bin]).T, aspect="auto", origin="lower")

        plt.plot(interp_track, linestyle="dotted", color="black")
          
        y, sr = librosa.load(utt_wav, sr=None)
          
        print("yin analyzing...")
        #f0_pyin, voiced_flag, voiced_probs = librosa.pyin(y,sr=sr, fmin=min_hz, fmax=max_hz, hop_length = round(sr/target_rate))
        print("yin done.")
      
        #print(len(f0_pyin), len(track))
        #track[track==0] = np.nan
        
        #plt.plot(f0_pyin*BINS_PER_HZ, color="red", label="pyin")
       
        plt.plot(track*BINS_PER_HZ, color="black", label="squeezer")
        plt.legend()
        plt.show()

    return (track/BINS_PER_HZ, interp_track/BINS_PER_HZ)



def _extract_to_file(utt_wav,min_hz=60, max_hz=500, voicing_thresh=0.3, target_rate=200, output_directory=None, wavelet=False, output_format="txt"):
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
        import torch
        torch.save(torch.from_numpy(if0), out_name+".interp.pt")
        torch.save(torch.from_numpy(f0), out_name+".f0.pt")
        if wavelet:
            torch.save(torch.from_numpy(cwt_mat),out_name+".cwt.pt")
            

def main():
    """
    Command line interface
    Example:
        extract pitch and cwt files to output_dir, two threads, numpy output, framerate compatible with fastpitch 22050hz:
        ```
        $ pitchsqueezer --nb_jobs 2 --frame_rate 86.1326 --min_hz 120 --max_hz 500 \\
        --wavelet -o output_dir/ -f npy mydata/female_wavs/ 

        $ pitchsqueezer --help
        
        usage: pitchsqueezer [-h] [-m MIN_HZ] [-M MAX_HZ] [-t VOICING_THRESH] [-r FRAME_RATE] [-j NB_JOBS] [-w] [-p] [-f {txt,pt,npy}] [-o OUTPUT_DIRECTORY] input

        ```
    """
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
            print("analyzing "+f)
            os.system("play -q "+f+ "&")
            f0, if0 = track_pitch(f,args.min_hz, args.max_hz, args.voicing_thresh, args.frame_rate, plot=True)
            if args.wavelet:
                f0_cwt(if0, plot=True)
                #f0_cwt(np.log(if0), plot=True)
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
    
    Parallel(n_jobs=args.nb_jobs)(delayed(_extract_to_file)(f, 
            args.min_hz, 
            args.max_hz, 
            args.voicing_thresh,
            args.frame_rate,
            args.output_directory,
            args.wavelet,
            args.output_format) for f in tqdm(input_files))

if __name__ == "__main__":
    sys.exit(main())