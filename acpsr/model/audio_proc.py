import sys
import numpy as np
from scipy import stats
from scipy.io import wavfile
from scipy.signal import spectrogram, medfilt
import librosa

default_fs = 16000

max_freq = 1000
min_freq = 75
# medium filter
wav_medium = 0
wav_median = 5
window_max = 20  # int or None

# Human speech frequency typically range from 75Hz to 250Hz,
# with extreme cases going towards 500Hz for certain consonants
# For safety, this is set to 75Hz-1000Hz
threshold_intensity = 0.075
# Total intensity for frequency range up to max_freq
threshold_timegap = 0.5
# Rule-based gap between segments, in seconds
threshold_time = 0
# minimum amount of time required for passing as a segment
threshold_min = 15
# minimum threshold value for intensity, absolute value


def wav_intensity(data, fs=default_fs, verbose=False):
    if isinstance(data, str):
        data, fs = load(data)
    if verbose:
        sys.stderr.write("Sampling rate:           %d\n" % (fs))
        sys.stderr.write("Recording length (sec):  %f\n" % (len(data) / fs))
        sys.stderr.write("Total num of samples:    %d\n" % (len(data)))

    # Calculate spectrogram
    freq, time, spec = spectrogram(data, fs)
    stop = -1
    for i in range(len(freq)):
        if freq[i] > max_freq:
            stop = i
            break
    start = -1
    for i in range(len(freq)):
        if freq[i] < min_freq:
            start = i
        else:
            break

    intensity = np.sum(spec[start:stop], 0)  # Sum of overall intensity
    return intensity, time


def calc_intensity_threshold(intensity):
    new_intensity = intensity
    # medium filter
    if wav_medium != 0:
        avg = wav_medium
        for i in range(1, avg + 1):
            new_intensity +=\
                np.pad(intensity, (i, 0), mode="constant")[:-i] +\
                np.pad(intensity, (0, i), mode="constant")[i:]
        new_intensity /= (1 + 2 * avg)

    # median filter
    if wav_median != 0:
        new_intensity = medfilt(new_intensity, kernel_size=wav_median)
    max_intensity = max(new_intensity)
    intensity_threshold = max(threshold_min, max_intensity * threshold_intensity)

    # Window max
    if window_max is not None:
        max_intensity = []
        for i in range(new_intensity.shape[0]):
            max_intensity.append(
                max(new_intensity[max(0, i - window_max): min(i + window_max, new_intensity.shape[0])])
            )
            max_intensity[i] = max(max_intensity[i], threshold_min)
        intensity_threshold = np.array(max_intensity) * threshold_intensity
    return intensity_threshold


def split_wavdata(data, fs=default_fs, verbose=True):
    if isinstance(data, str):
        data, fs = load(data)
    intensity, time = wav_intensity(data, fs=fs, verbose=verbose)

    if verbose:
        sys.stderr.write("Threshold Intensity    : %f\n" % (threshold_intensity))
        sys.stderr.write("Medium filter kernal   : %d\n" % (wav_medium))
        sys.stderr.write("Median filter kernal   : %d\n" % (wav_median))

    intensity_thresholds = calc_intensity_threshold(intensity)
    segments = []
    last_activation = -1
    for i in range(len(intensity)):
        if isinstance(intensity_thresholds, np.ndarray):
            intensity_threshold = intensity_thresholds[i]
        else:
            intensity_threshold = intensity_thresholds
        if intensity[i] >= intensity_threshold and \
                time[i] - last_activation > threshold_timegap:
            # Intensity above threshold, gap between previous great
            # create new segment
            segments.append((i, i))
            last_activation = time[i]
        elif intensity[i] >= intensity_threshold and \
                time[i] - last_activation <= threshold_timegap:
            # Intensity above threshold, gap between previous small
            # continue segment
            segments[-1] = segments[-1][0], i
            last_activation = time[i]
        elif intensity[i] < intensity_threshold and \
                time[i] - last_activation <= threshold_timegap:
            # Intensity below threshold, gap between previous small
            # continue segment
            segments[-1] = segments[-1][0], i

    # Right now we have segments in seconds, calculate start and end sample index
    for i in range(len(segments)):
        t_s, t_e = time[segments[i][0]], time[segments[i][1]]
        s_s = int(max(t_s - 0.1, 0) * fs)
        s_e = min(int((t_e + 0.1) * fs), len(data))
        if verbose:
            sys.stderr.write("Clip recognised   : %d - %d\n" % (s_s, s_e))

        segments[i] = data[s_s:s_e]

    # Filtering out segments that are too short
    segments = [seg for seg in segments if len(seg) > threshold_time * fs]

    total_sample_segments = sum([len(seg) for seg in segments])
    if verbose:
        sys.stderr.write("Total num of segments:   %d\n" % (len(segments)))
        sys.stderr.write("Total time of segments:  %f\n" % (total_sample_segments / fs))
        sys.stderr.write("Keep percentage (%%):     %.2f\n" % (total_sample_segments / len(data) * 100))
    return segments, fs


def load_audio_file(filename, convert=False):
    if not isinstance(filename, str):
        raise TypeError("Parameter filename must be an instance of str")
    if len(filename) <= 4:
        raise ValueError("Invalid filename: " + filename)
    fs, data = None, None
    if filename[-4:] == ".wav":
        fs, data = wavfile.read(filename)
    elif filename[-4:] == ".mp3":
        if convert:
            from pydub import AudioSegment
            tgt = filename[:-4] + ".wav"
            # Load MP3 and resample to default sampling rate
            sound = AudioSegment.from_mp3(filename).set_frame_rate(default_fs)
            sound.export(tgt, format="wav")
            fs, data = wavfile.read(tgt)
        else:
            data, fs = librosa.load(filename, sr=default_fs)
    return data, fs


def plot_audio_data(data, fs=default_fs, spectrogram=False, intensity=False):
    if isinstance(data, str):
        data, fs = load_audio_file(data)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(data)
    plt.xlabel("Sample Index (FS=%d)" % (fs))
    plt.ylabel("Amplitude")
    plt.title("Wavform")
    plt.show()

    if spectrogram:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.specgram(data, Fs=fs)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title("Spectrogram")
        plt.show()
    if intensity:
        import matplotlib.pyplot as plt
        intensity, time = wav_intensity(data, fs)
        plt.figure()
        plt.plot(intensity)
        plt.xlabel("Spectrogram Index (FS=%d)" % (fs))
        plt.ylabel("Amplitude")
        threshold = calc_intensity_threshold(intensity)
        if not isinstance(threshold, np.ndarray):
            print(threshold)
            plt.axhline(y=threshold, color='r', linestyle='-')
            plt.title("Spectrogram Frequency Intensity (%dHz-%dHz), Threshold (%f)" % (min_freq, max_freq, threshold))
        else:
            plt.plot(threshold, color='r', linestyle='-')
            plt.title("Spectrogram Frequency Intensity (%dHz-%dHz)" % (min_freq, max_freq))

        plt.show()


def generate_noise(ms, max_amp=0.001, fs=default_fs):
    amplitude = 11
    noise = stats.truncnorm(-1, 1, scale=min(2**16, 2**max_amp)).rvs(ms * fs // 1000)
    noise = noise.astype(np.int16)
    return noise


def export_audio_data(filename, data, fs=default_fs):
    if not isinstance(filename, str):
        raise TypeError("filename must be an instance of str")
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be an instance of np.ndarray")
    if not isinstance(fs, int):
        raise TypeError("fs must be an instance of int")
    wavfile.write(filename, fs, data)


split = split_wavdata
load = load_audio_file
export = export_audio_data
plot = plot_audio_data
noise = generate_noise
