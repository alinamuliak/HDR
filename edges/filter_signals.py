import numpy as np
import random
from scipy.signal import csd
from time import time
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.integrate import simps

from read_json_results import snr_on_second


def scale(sgnl):
    mx, mn = np.max(sgnl), np.min(sgnl)
    return sgnl / (mx - mn)


def cnvd(sgnl, fltr):
    fltd = np.convolve(scale(sgnl), scale(fltr), mode='same')
    fltd = scale(fltd)
    return fltd


def consec_cnvd(signals, idx):
    return cnvd(signals[idx], signals[idx + 1])


def get_1st_order_pool(signals, num):
    pool = []
    ids = list(range(len(signals) - 1))
    random.shuffle(ids)
    for _, i in zip(range(num), ids):
        pool.append(consec_cnvd(signals, i))
    return pool


def get_coherence(x, y) -> tuple:
    """
    Calculate coherence in a following way:
    Cxy = |Pxy|^2 / sqrt(Pxx * Pyy),
    where P__ - is a  cross power spectral density using Welch method.
    """
    f, Pxy = csd(x, y, fs=30)
    Pxy = np.array(Pxy)
    denominator = np.sqrt(np.array(csd(x, x, fs=30)[1]) * np.array(csd(y, y, fs=30)[1]))
    Cxy = np.abs(Pxy) ** 2 / denominator
    return f, Cxy


def read_signals_data(signals_path, timestamps_path):
    all_signals, all_timestamps = [], []

    with open(signals_path, "rb") as all_signals_file:
        try:
            while True:
                signal = np.load(all_signals_file)
                all_signals.append(signal)
        except Exception as err:
            print("Signals reading finished.")

    with open(timestamps_path, "rb") as all_timestamps_file:
        try:
            while True:
                all_timestamps.append(np.load(all_timestamps_file))
        except Exception as err:
            print("Signals' timestamps reading finished.")
    return all_signals, all_timestamps


def choose_best(signals, good_t=20, max_iter=5, show_time=False):
    coherence_time = []
    good_signals = []
    not_certain_signals = []
    iteration = 0
    coherence_values = {0:0}
    while iteration < max_iter and min(list(coherence_values)) < good_t:
        random.shuffle(signals)
        pairs = []
        coherence_values = {}
        for i in range(0, len(signals) - 1, 2):
            pairs.append((signals[i], signals[i+1]))

            if show_time:
                coherence_start_time = time()
            f, coh = get_coherence(signals[i], signals[i+1])
            if show_time:
                coherence_time.append(time() - coherence_start_time)

            informative_indexes = (f * 60 >= 5) & (f * 60 <= 30)
            max_peak = max(coh[informative_indexes])
            coherence_values[max_peak] = pairs[-1]  # append
        good_signals = []
        quantile = np.quantile(list(coherence_values.keys()), 0.25)
        for peak in coherence_values:
            if quantile < peak:
                good_signals.append(coherence_values[peak][0])
                good_signals.append(coherence_values[peak][1])
        iteration += 1
        signals = good_signals

    if show_time:
        print(f"\navg coherence calculation time: {np.average(coherence_time)}")

    if good_signals:
        return list(set(tuple(good_signals[i]) for i in range(len(good_signals))))
    return set(tuple(not_certain_signals[:10]))


def filter_signals(signals_path, timestamps_path, show_time=False):
    if show_time:
        read_start = time()
    signals, ts = read_signals_data(signals_path, timestamps_path)

    if show_time:
        start = time()
    # pooled = get_1st_order_pool(signals[signals_idx], 50)
    pooled = signals[0]  # todo: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    filtered_signals = choose_best(pooled)
    if show_time:
        filter_time = time()
        print(f"total filter time: {filter_time - start}\n\nread time: {start - read_start}")

    return filtered_signals


def peak_normalized_power(spectrum, peak_start, peak_end):
    """
    :param spectrum: array with the spectrum power values
    :param peak_start: index of the start of the peak
    :param peak_end: index of the end of the peak
    :return normalized power: normalized power value
    Given function computes the normalized power of the peak. The normalized power of the peak is defined as the
    ratio between power of the peak and the power of the whole spectrum.
    """
    peak_power = simps(spectrum[peak_start:peak_end + 1], dx=0.1)

    total_power = simps(spectrum, dx=0.1)

    normalized_power = peak_power / total_power

    return normalized_power


def snr_freq_domain(sig, ts, gt_value, margin_width=2, highcut=60, plot=False):
    xf, yf = LombScargle(ts, sig).autopower()

    xf *= 60
    end_freq_indx = np.where(xf >= highcut)[0][0]
    xf, yf = xf[:end_freq_indx], yf[:end_freq_indx]

    bounds_start, bounds_end = gt_value - margin_width, gt_value + margin_width

    bounds_start_index = np.argmin([abs(bounds_start - x) for x in xf])
    bounds_end_index = np.argmin([abs(bounds_end - x) for x in xf])

    norm_power = peak_normalized_power(yf, bounds_start_index, bounds_end_index)

    if plot:
        gt_index = np.argmin([abs(gt_value - x) for x in xf])
        idx_margin_area = np.logical_and(xf >= xf[bounds_start_index], xf <= xf[bounds_end_index])

        fig, ax = plt.subplots()

        ax.plot(xf, yf)
        ax.plot(xf[gt_index], yf[gt_index], "x", color="green", markeredgewidth=5)
        ax.fill_between(xf, yf, where=idx_margin_area, color='skyblue')

        text = f"Norm power: {round(norm_power, 3)}"
        ax.text(0.9, 0.9, text, verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes,
                color='green', fontsize=15)

        # plt.title(f"Peridiogram of motion signal #{i} with bounds = {margin_width} BPM")
        plt.show()

    return norm_power


if __name__ == "__main__":
    signals_filenames = [f'motion_signals_examples/{file}' for file in ['all_motion_signals_cohface_1_1.npy',
                                                                        'all_motion_signals_cohface_3_1.npy',
                                                                        'all_motion_signals_ss_mobile_subject4_record7.npy',
                                                                        'all_motion_signals_ss_mobile_subject7_record7.npy',
                                                                        'all_motion_signals_ss_web_5_20.npy',
                                                                        'all_motion_signals_ss_web_8_15.npy']]
    ts_filenames = [f'motion_signals_examples/{file}' for file in ['all_motion_signals_timestamps_cohface_1_1.npy',
                                                                    'all_motion_signals_timestamps_cohface_3_1.npy',
                                                                    'all_motion_signals_timestamps_ss_mobile_subject4_record7.npy',
                                                                    'all_motion_signals_timestamps_ss_mobile_subject7_record7.npy',
                                                                    'all_motion_signals_timestamps_ss_web_5_20.npy',
                                                                    'all_motion_signals_timestamps_ss_web_8_15.npy']]
    snr_stat_filenames = [f'motion_signals_examples/{file}' for file in ['signal_snr_statistics_3_1_1.json',
                                                                          'signal_snr_statistics_3_3_1.json',
                                                                          'signal_snr_statistics_3_Subject_4_record7.json',
                                                                          'signal_snr_statistics_3_Subject_7_record7.json',
                                                                          'signal_snr_statistics_3_5_20.json',
                                                                          'signal_snr_statistics_3_8_15.json']]
    gt_values = [17, 12, 15, 15, 20, 15]
    result_filenames = [f'results/{file}.txt' for file in ['cohface_1_1',
                                                           'cohface_3_1',
                                                           'ss_mobile_4_7',
                                                           'ss_mobile_7_7',
                                                           'ss_web_5_20',
                                                           'ss_web_8_15']]

    for i in range(len(signals_filenames)):
        with open(result_filenames[i], 'a') as f:
            f.write(f'FILE:\t\t\t\t{signals_filenames[i]}\n')
            signals, ts = read_signals_data(signals_filenames[i],
                                            ts_filenames[i])

            for second in range(len(ts)):
                filtered = choose_best(signals[second])
                snrs = []
                for signal in filtered:
                    snr = snr_freq_domain(signal, ts[second], gt_values[i])
                    snrs.append(round(snr, 3))

                coherence = sorted(snrs, reverse=True)[:20]
                autocorr = sorted(snr_on_second(snr_stat_filenames[i], second), reverse=True)

                f.write(f'SECOND:\t\t\t\t{second}\n')
                f.write(f'COHERENCE:\t\t\t{coherence}\n')
                f.write(f'AUTOCORRELATION:\t{autocorr}\n\n')
