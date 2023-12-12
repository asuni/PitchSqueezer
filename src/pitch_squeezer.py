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



plt.rcParams.update({'font.size': 6})



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



def _hp_filter(sig, cutoff_frequency=60, order=4):
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

def _stack_f0(spec, voiced_frames, min_hz, max_hz):
    max_i= np.argmax(spec[:, min_hz:max_hz], axis=1)+min_hz
    max_vals = np.log(spec[np.arange(spec.shape[0]), max_i])
    mag_mean= np.mean(max_vals[voiced_frames])
    mag_std= np.std(max_vals[voiced_frames])
    max_i[np.invert(voiced_frames)]=0
    threshold = mag_mean-mag_std*0.5
    for i in range(0, spec.shape[0]):
        cand = max_i[i]
        if cand > 2*min_hz:
            if np.log(spec[i,int(cand/2)]) > threshold or spec[i, int(cand/2)] > spec[i, cand] * 0.2:
                spec[i, int(cand/2)]+=spec[i, cand]
    return spec


def _apply_varying_window(spec, unvoiced_frames, win_mul = 1.5, min_hz=50, max_hz=500, bins_per_hz=1):
        
    pitch = _get_max_track(spec, unvoiced_frames, min_hz, max_hz).astype('float')
    pitch = _trim_voice_boundaries(pitch, 2)
    pitch[pitch==0] = np.nan
    pitch = scipy.signal.medfilt(pitch,5)
    std = np.nanstd(pitch)
    pitch = _interpolate_zeros(pitch, 'linear')
    pitch = _smooth(pitch, 31)
   
    for i in range(0, spec.shape[0]):
       
        l_window = scipy.signal.windows.gaussian(int(pitch[i]*2.),std*win_mul)
        r_window = scipy.signal.windows.gaussian(int(max_hz-pitch[i]+1)*2, std*win_mul)
       
        window = np.concatenate((l_window[:len(l_window)//2], r_window[len(r_window)//2:]))
        spec[i,:len(window)]*=window+np.mean(spec[i,:max_hz])
        
    return spec

def _add_slope(spec, min_hz=50, max_hz=500, steepness=1.):
    orig_energy = np.sum(spec[:,min_hz:max_hz])
    #increasing spectral slope, in order to make fundamental stand out more
    if min_hz<1:min_hz = 1
    spec[:, np.arange(min_hz, max_hz)] /=  np.arange(min_hz,max_hz) ** steepness
    spec[:, np.arange(min_hz, max_hz)] *= orig_energy / np.sum(spec[:, np.arange(min_hz, max_hz)])
    #spec[]
    return spec

def _get_max_track(spec, unvoiced_frames =[],min_hz=50, max_hz=500):
    f0 = np.argmax(spec[:, int(min_hz):int(max_hz)], axis=1)+min_hz 
    f0[unvoiced_frames] = 0
    return f0

def _get_viterbi_track(spec, voiced_frames = [], min_hz_bin=50, max_hz_bin=500, penalty=1, subsample=1):
    # narrow search space
    if len(voiced_frames) > 0:
        voiced_f0 = np.argmax(spec[voiced_frames, min_hz_bin:max_hz_bin], axis=1)+min_hz_bin
        min_hz_bin,max_hz_bin = np.min(voiced_f0), np.max(voiced_f0)

    # change frequencies to semitones to make penalty more perceptually valid
    scales = np.arange(min_hz_bin, max_hz_bin, subsample)
    scales = _hz_to_semitones(scales, min_hz_bin)
   
    # forward-backward search, optional speed up by decimating the spectrogram
    if subsample > 1:
        spec_sub = spec[:,::subsample].copy()
        min_hz_bin = int(min_hz_bin/subsample)
        max_hz_bin = min_hz_bin + len(scales)
        track = extract_ridges(spec_sub[:,min_hz_bin:max_hz_bin].T,scales, penalty=penalty, n_ridges=1, transform="fft")
    else:
        track = extract_ridges(spec[:,min_hz_bin:max_hz_bin].T,scales, penalty=penalty, n_ridges=1, transform="fft")  
    
    track = np.array(track).flatten()*subsample+(min_hz_bin*subsample)
    
    return track


def _trim_voice_boundaries(arr, n):
    voiced_indices = np.where(arr)[0]
    trimmed = np.array(arr)
    for idx in np.split(voiced_indices, np.where(np.diff(voiced_indices) != 1)[0] + 1):
        trimmed[idx[:n]] = False
        trimmed[idx[-n:]] = False
    
    return trimmed

def _process_outliers(pic, pitch, mode="shift"):
  
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    mean_track = _trim_voice_boundaries(pitch, 2)
  
    mean_track = scipy.signal.medfilt(mean_track, 5)

    #calc running mean and std
    mean_track[mean_track==0]=np.nan
    N = 100
    if N > len(pitch)/2:
        #N = int(len(fixed)//4)
        return pic, pitch
   
    idx = np.arange(N) + np.arange(len(mean_track)-N)[:,None]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice.')
        mean_track = np.nanmedian(mean_track[idx],axis=1)

    #mean_track = scipy.signal.medfilt(mean_track, 31)
    mean_track = np.pad(mean_track, N//2,'edge')
    mean_track = _interpolate_zeros(mean_track, 'linear')
    mean_track = _smooth(mean_track, 50)
    mean_track[pitch==0] = 0
   
    if mode == "remove":
        pitch[pitch<mean_track*0.6]=0
        pitch[pitch>mean_track*1.9]=0 
    elif mode == "shift":
        """
        plt.plot(pitch)
        plt.plot(mean_track)
        plt.plot(mean_track*0.75, linestyle="dashed")
        plt.plot(mean_track*1.5, linestyle="dashed")
        plt.show()
        print(pic.shape)
        """
        cond1 = pitch < (mean_track * 0.6)
        cond2 = pitch > mean_track * 1.9
      
        #pic[cond1, (pitch[cond1]).astype('int')]*=0 #0.01 #10 #pitch[cond1]*0.1
        #pic[cond2, (pitch[cond2]).astype('int')]*=0 #.01 #10 #=pitch[cond2]*0.1
        pic[cond1, (pitch[cond1]*2).astype('int')]*=10 #pitch[cond1]*0.1
        pic[cond2, (pitch[cond2]/2).astype('int')]*=10 #pitch[cond2]*0.1 #10 #=pitch[cond2]*0.1
        print(pic.shape)
    return pic, pitch

def _remove_outliers(pitch, mode="remove"):
  
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    mean_track = _trim_voice_boundaries(pitch, 2)
  
    mean_track = scipy.signal.medfilt(mean_track, 5)

    #calc running mean and std
    mean_track[mean_track==0]=np.nan
    N = 100
    if N > len(pitch)/2:
        #N = int(len(fixed)//4)
        return pitch
   
    idx = np.arange(N) + np.arange(len(mean_track)-N)[:,None]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        warnings.filterwarnings('ignore', r'Degrees of freedom <= 0 for slice.')
        mean_track = np.nanmedian(mean_track[idx],axis=1)

    #mean_track = scipy.signal.medfilt(mean_track, 31)
    mean_track = np.pad(mean_track, N//2,'edge')
    mean_track = _interpolate_zeros(mean_track, 'linear')
    mean_track = _smooth(mean_track, 50)

    if mode == "remove":
        pitch[pitch<mean_track*0.6]=0
        pitch[pitch>mean_track*2.5]=0 
    elif mode == "shift":
        for i in range(2):
            pitch[pitch<mean_track*0.55]*=2 #std_track*2]=0.
            pitch[pitch>mean_track*1.95]/=2
    return pitch

def _remove_bias(spec, max_hz=None, percentile = 5):
    if max_hz is None:
        max_hz = spec.shape[1]
    e = np.sum(spec[:,:max_hz], axis=1)
    threshold = np.percentile(e, percentile)
    indices = np.where(e <= threshold)
    bias_spectrum = np.mean(spec[indices, :max_hz], axis=1).flatten()
    bias_spectrum = _smooth(bias_spectrum, 20)
    mean_val = np.mean(bias_spectrum)
    spec[:, :max_hz] -= bias_spectrum
    return spec

def _get_voicing(acorr1, short_win_fft, pic, min_hz_bin, max_hz_bin, voicing_thresh):
  
    e1 = _norm(np.log(np.sum(acorr1, axis=1)+1.))
    e2 = _norm(np.log(np.sum(short_win_fft[:,min_hz_bin:max_hz_bin], axis=1)+1.))
    e3 = _norm(np.log(np.max(pic[:,min_hz_bin:max_hz_bin], axis=1)+0.0001) - \
        np.log(np.median(pic[:,min_hz_bin:max_hz_bin], axis=1)+0.0001))
   
    voicing_strength = e1*0.5+e2+e3
    voicing_strength-=np.percentile(voicing_strength, 10)
    voicing_strength[voicing_strength<0] = 0
    unvoiced_frames  = voicing_strength < voicing_thresh
    
    unvoiced_frames = scipy.signal.medfilt(unvoiced_frames.astype('int'), 5).astype('bool')
    
    voiced_frames =  np.invert(unvoiced_frames)
    return voicing_strength, voiced_frames, unvoiced_frames

def _norm(params):
    return (params-np.min(params))/np.ptp(params)

def _plt(spec, uv_frames, min=0, max=500, ax=None, title=""):
    ax.imshow(np.log(spec[:,int(min):int(max)]).T, aspect="auto", origin="lower",cmap="viridis") #,interpolation="None")
    track = _get_max_track(spec, uv_frames, max_hz=max).astype('float')
    track[uv_frames]=np.nan
    ax.plot(track, color="white", linestyle="dotted")
    ax.set_title(title, loc="left", x=0.02, y=0.0, color="white")


# main function
def track_pitch(utt_wav ,min_hz=60,max_hz=500, voicing_thresh=0.5,frame_rate=100, viterbi=True, plot=False):
    """
    Extracts f0 track from speech .wav file using synchrosqueezed spectrogram and frequency domain autocorrelation.
    Args:
        utt_wav (str): path to the audio file
        min_hz (int): minimum f0 value to consider
        max_hz (int): maximum f0 value to consider
        voicing_thresh (float): voicing decision tuner, 0.:all voiced, 1.:nothing voiced
        frame_rate (float): number of pitch values per second to output (sr/frame_length_in_samples)
        viterbi (bool): forward-backward search for smooth path 
        plot (bool): visualize the analysis process

    Returns:
        tuple containing 2 arrays with length  np.floor(len(wav)/(sr//frame_rate)+1.) 
        - track (np.array): array containing f0 values, unvoiced frames=0. 
        - interp_track (np.array)): array containing f0 values with unvoiced gaps filled using interpolation

    Example:
        ```
        import pitch_squeezer as ps
        f0, if0 = ps.track_pitch("test.wav", frame_rate=200)
        lf0_cwt = ps.f0_cwt(np.log(if0))
        ```
    """
    
    # some internal constants, could be user params also
    SR = 4000.0          # sample rate should be high enough for spectral autocorrelation (3 harmonics form max_hz)
    INTERNAL_RATE = 100  # frames per second, 100 for speed, >=200 for accuracy
    BINS_PER_HZ = 1.     # determines the frequency resolution of the generated track, slows down rapidly if increased > 2
    SPEC_SLOPE = 1.5     # adjusting slope steeper will emphasize lower harmonics
    ACORR_WEIGHT = 1.5    #
    VITERBI_PENALTY = 3*INTERNAL_RATE*0.01/BINS_PER_HZ  # larger values will provide smoother track but might cut through fast moving peaks
    MIN_VAL = 1.0e-6
    SOFT_WINDOW_STD = 1.5     # stds. determines how tightly the spectrum is filtered around running mean
    min_hz_bin = int(min_hz * BINS_PER_HZ)
    max_hz_bin = int(max_hz * BINS_PER_HZ)
    orig_sr = librosa.get_samplerate(utt_wav)
   
    outlier_removal=True

    # read wav file, downsample to 4000Hz, highpass filter to get rid of hum, and normalize
    sig, orig_sr = librosa.load(utt_wav, sr=None)
    orig_sig_len = len(sig) # needed for target frame rate conversion
    sig = librosa.resample(sig, orig_sr=orig_sr, target_sr=SR)
    
    sig = _hp_filter(sig, cutoff_frequency=min_hz-10)
    sig = (sig-np.mean(sig)) / np.std(sig) 
   
    # dither
    sig+=0.01*np.random.normal(size=len(sig))

    #  stfts on the signal
    frame_shift = int(SR//INTERNAL_RATE)
    ssq_fft ,fft2, *_ = ssq_stft(sig ,n_fft = int(SR*BINS_PER_HZ), win_len=int(SR/4),hop_len=frame_shift)
    ssq_fft = abs(ssq_fft).T # personal preference for (time, hz) shape
    short_win_fft, short_win_fft2,*_ = ssq_stft(sig, n_fft=int(SR*BINS_PER_HZ),win_len=int(SR/20),hop_len=frame_shift)
    short_win_fft = abs(short_win_fft).T

    if plot:
        fig, ax = plt.subplots(6, 1, sharex=True, sharey=True)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0.1, wspace = 0)
        tmp_voicing_strength = np.sum(abs(short_win_fft2.T)[:,min_hz:max_hz], axis=1)
        unvoiced_frames = tmp_voicing_strength < voicing_thresh*1000
        _plt(abs(fft2).T, unvoiced_frames, max=max_hz_bin, ax=ax[0], title="fft")

    # pic will be our time-frequency image that is manipulated by subsequent 
    # methods to make fundamental frequency stand out in the spectra
    pic = scipy.ndimage.gaussian_filter(ssq_fft,[0.5,1.*BINS_PER_HZ])
    pic = _remove_bias(pic)
    pic[pic<MIN_VAL] = MIN_VAL

    if plot:
        _plt(pic, unvoiced_frames, max = max_hz_bin, ax=ax[1], title="synchro-squeezing")

    # frequency domain autocorrelation, up to 3 * max_hz to capture at least 3 harmonics
    length = min(max_hz_bin*3,int(SR*BINS_PER_HZ/2))
    acorr1 = librosa.autocorrelate(pic[:,:length], max_size=length, axis=1)

    
    # add strongest autocorrelation frequency to spec, helps with weak or missing fundamental
    if viterbi:
        acorr_f0 = _get_viterbi_track(acorr1, min_hz_bin=min_hz_bin, max_hz_bin=max_hz_bin, penalty=1, subsample=4)
    else:
        acorr_f0 = np.argmax(acorr1[:, min_hz_bin:max_hz_bin], axis=1)+min_hz_bin    
    

    for i in range(-2,2):
        pic[np.arange(pic.shape[0]), acorr_f0+i] += acorr1[np.arange(pic.shape[0]), acorr_f0+i]*ACORR_WEIGHT
    
    # estimate voicing strength 
    voicing_strength, voiced_frames, unvoiced_frames = \
        _get_voicing(acorr1, short_win_fft, pic, min_hz_bin, max_hz_bin, voicing_thresh)
   
    
    # multiply spec with the whole correlogram, suppresses non-harmonic stuff
    pic[:, :length] *=acorr1
    pic = _norm(pic)
    
    if plot:
        _plt(pic, unvoiced_frames, max = max_hz_bin, ax=ax[2], title="+freq autocorrelation")
       
    
    # dampen higher frequencies to reduce octave jumps up
    pic = _add_slope(pic, min_hz=0, max_hz=max_hz_bin, steepness=SPEC_SLOPE)
    
    # reassign energy from h2 to h1 
    pic = _stack_f0(pic, voiced_frames, min_hz_bin, max_hz_bin)
   
    if plot:
        _plt(pic, unvoiced_frames, max = max_hz_bin, ax=ax[3], title="+skew spectrum + harmonic reassignment")
        
    # softy constrain the hypothesis space by windowing around current pitch trajectory 
    pic = _apply_varying_window(pic, unvoiced_frames, SOFT_WINDOW_STD, min_hz_bin, max_hz_bin, BINS_PER_HZ)
   
    if plot:
        _plt(pic, unvoiced_frames, max = max_hz_bin, ax=ax[4], title="+window around running mean + std ")
    
    # get final pitch track
    if viterbi:
        track = _get_viterbi_track(pic, voiced_frames, min_hz_bin, max_hz_bin, VITERBI_PENALTY, subsample=2)
        
        track[unvoiced_frames]= 0
        #ax[5].plot(track, color="red") #, linestyle="dotted")
        #pic, track = _process_outliers(pic,track.astype('float'), mode="shift")
        #track = _get_viterbi_track(pic, voiced_frames, min_hz_bin, max_hz_bin, VITERBI_PENALTY, subsample=1)
        track = track.astype('float')
    else:
        track =_get_max_track(pic, unvoiced_frames, min_hz, max_hz).astype('float')
        track = scipy.signal.medfilt(track, 3)

    track[unvoiced_frames] = 0
    if np.all(track == 0):
        print(utt_file+" all unvoiced!.")
        return (track, track)
    
    if plot:
        ax[5].imshow(np.log(pic[:,:max_hz_bin]).T, aspect="auto", origin="lower")
        ax[5].plot(track, color="white") #, linestyle="dotted")
        ax[5].set_title("+viterbi", loc="left",x=0.02, y=0., color="white")
        plt.show()

 
    # fill unvoiced gaps and postprocess 
    if outlier_removal:
        track = _remove_outliers(track, mode="remove")

    unvoiced_frames[track==0] = True
    interp_track = _trim_voice_boundaries(track, 1)
    interp_track = _interpolate_zeros(interp_track, method='akima')
    interp_track = scipy.signal.medfilt(interp_track, 3)
    interp_track = _smooth(interp_track, 3)
    track = np.array(interp_track)
    track[unvoiced_frames] = 0
    # convert to target frame rate         
    n_target_frames = np.floor(orig_sig_len/(orig_sr//frame_rate)+1.).astype('int')
    track = _apply_target_rate(track, INTERNAL_RATE, n_target_frames)
    interp_track = _apply_target_rate(interp_track, INTERNAL_RATE, n_target_frames)
    unvoiced_frames = _apply_target_rate(unvoiced_frames.astype('int'), INTERNAL_RATE, n_target_frames).astype('bool')
    track[unvoiced_frames] = 0 

    if plot:
        plt.figure(figsize=(12,4))
        if frame_rate == INTERNAL_RATE:
            ssq_fft = scipy.ndimage.gaussian_filter(ssq_fft,[1,1*BINS_PER_HZ])
            plt.imshow(np.log(ssq_fft[:,:max_hz_bin]+MIN_VAL).T, aspect="auto", origin="lower")
        plt.plot(interp_track, linestyle="dotted", color="white",alpha=0.5)
        plot_track = np.array(track)
        plot_track[track==0] = np.nan

        y, sr = librosa.load(utt_wav, sr=None)
        f0_pyin, voiced_flag, voiced_probs = librosa.pyin(y,sr=sr, fmin=min_hz, fmax=max_hz, hop_length = int(sr/frame_rate))
        ax[5].plot(f0_pyin*BINS_PER_HZ, color="red", label="pyin",linestyle="dotted")
        ax[5].legend()
        plt.plot(plot_track, color="white", label="squeezer")#,linestyle="dotted")
        #plt.plot(voicing_strength*50, color="yellow",label="voicing_strength")
        plt.legend()
        plt.show()

    return (track/BINS_PER_HZ, interp_track/BINS_PER_HZ)



def _extract_to_file(utt_wav,min_hz=60, max_hz=500, voicing_thresh=0.3, frame_rate=200, output_directory=None, wavelet=False, output_format="txt"):
    f0, if0 = track_pitch(utt_wav, min_hz, max_hz, voicing_thresh, frame_rate)
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

        # check parameters by visual examination
        $ pitchsqueezer wav_dir/ --plot

        # check usage and default values 
        $ pitchsqueezer --help
        
        usage: pitchsqueezer [-h] [-m MIN_HZ] [-M MAX_HZ] [-t VOICING_THRESH] [-r FRAME_RATE] [-j NB_JOBS] [-w] [-p] [-f {txt,pt,npy}] [-o OUTPUT_DIRECTORY] input
        ...
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
    import cProfile
    
    if args.plot:
        import signal
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        import random
        random.shuffle(input_files)
        for f in input_files:
            print("analyzing "+f+".  (ctrl-c to quit.)")
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