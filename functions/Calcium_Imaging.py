"""
Calcium Imaging Signal Processing Toolkit

This module provides functions for preprocessing, smoothing, baseline correction,
deconvolution, spike detection, and peak feature extraction from calcium imaging traces.
It supports input as pandas DataFrames with a 'Time' column and traces in remaining columns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter, find_peaks, peak_widths, argrelmin
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
from caiman.source_extraction.cnmf import deconvolution

# ----------------------------------
# SCALING FUNCTION
# ----------------------------------
def ScaleTraces(df):
    """
    Scales calcium traces in the DataFrame to [0, 1] using Min-Max normalization.

    Args:
        df (pd.DataFrame): Input DataFrame with 'Time' column and cell traces.

    Returns:
        pd.DataFrame: Scaled DataFrame with same structure.
    """
    time_column = df['Time']
    cell_traces = df.drop(columns=['Time'])
    flattened_data = cell_traces.values.flatten().reshape(-1, 1)
    scaled_data = MinMaxScaler().fit_transform(flattened_data).reshape(cell_traces.shape)
    scaled_df = pd.DataFrame(scaled_data, columns=cell_traces.columns)
    scaled_df.insert(0, 'Time', time_column)
    return scaled_df

# ----------------------------------
# SMOOTHING FUNCTION
# ----------------------------------
def smooth_signal(df, window_length=11, polyorder=2):
    """
    Applies Savitzky-Golay filter to smooth each cell trace.

    Args:
        df (pd.DataFrame): DataFrame with 'Time' column and cell traces.
        window_length (int): Window size for filter.
        polyorder (int): Polynomial order for filter.

    Returns:
        pd.DataFrame: Smoothed DataFrame.
    """
    smoothed_df = df.copy()
    for col in df.columns[1:]:
        smoothed_df[col] = savgol_filter(df[col], window_length, polyorder)
    return smoothed_df

# ----------------------------------
# BASELINE CORRECTION (ALS)
# ----------------------------------
def baseline_als(y, lam=1e6, p=0.01, niter=10):
    """
    Asymmetric least squares baseline correction.

    Args:
        y (np.array): Input signal.
        lam (float): Smoothing parameter.
        p (float): Asymmetry.
        niter (int): Number of iterations.

    Returns:
        np.array: Estimated baseline.
    """
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = diags(w, 0, shape=(L, L))
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def apply_als_baseline(df, lam=1e6, p=0.01):
    """
    Applies ALS baseline correction across all cell traces.

    Args:
        df (pd.DataFrame): DataFrame with 'Time' and cell traces.
        lam (float): ALS smoothing parameter.
        p (float): ALS asymmetry parameter.

    Returns:
        pd.DataFrame: Baseline-corrected traces.
    """
    corrected_df = df.copy()
    for col in tqdm(df.columns[1:]):
        baseline = baseline_als(df[col].values, lam=lam, p=p)
        corrected_df[col] = df[col] - baseline
    return corrected_df

# ----------------------------------
# DECONVOLUTION
# ----------------------------------
def apply_deconvolution(df, p=1):
    """
    Applies constrained deconvolution (FOOPSI) using CaImAn to estimate spike trains.

    Args:
        df (pd.DataFrame): DataFrame with 'Time' and calcium traces.
        p (int): Order of autoregressive system.

    Returns:
        dict: Dictionary with inferred spikes for each cell.
    """
    inferred_spikes = {}
    for col in df.columns[1:]:
        signal = df[col].values
        spike_train = deconvolution.constrained_foopsi(fluor=signal, p=p)
        inferred_spikes[col] = spike_train[0]
    return inferred_spikes

# ----------------------------------
# PEAK DETECTION
# ----------------------------------
def detect_peaks_and_features(df, prominence=0.01, distance=1):
    """
    Detects peaks and calculates their features from each cell trace.

    Args:
        df (pd.DataFrame): Input DataFrame.
        prominence (float): Required prominence of peaks.
        distance (int): Minimum distance between peaks.

    Returns:
        dict: Dictionary with peak features for each trace.
    """
    results = {}
    for col in df.columns[1:]:
        peaks, properties = find_peaks(df[col], prominence=prominence, distance=distance)
        peak_amplitudes = df[col].iloc[peaks].values
        widths_res = peak_widths(df[col], peaks, rel_height=0.5)
        results[col] = {
            'Peaks': peaks,
            'Amplitudes': peak_amplitudes,
            'Prominences': properties['prominences'],
            'Widths': widths_res[0],
            'Left_Bases': widths_res[2],
            'Right_Bases': widths_res[3]
        }
    return results

# ----------------------------------
# UTILITY: DECONVOLUTION DF COMBINATION
# ----------------------------------
def update_with_deconvolved_spikes(corrected_df, inferred_spikes):
    """
    Appends deconvolved spikes to the corrected DataFrame.

    Args:
        corrected_df (pd.DataFrame): Baseline-corrected DataFrame.
        inferred_spikes (dict): Inferred spike dictionary.

    Returns:
        pd.DataFrame: Updated DataFrame with spike columns.
    """
    deconvolved_spikes_df = pd.DataFrame(inferred_spikes)
    return pd.concat([corrected_df, deconvolved_spikes_df.add_suffix('_spikes')], axis=1)

def create_spikes_df(corrected_df, inferred_spikes):
    """
    Combines time and inferred spikes into a single DataFrame.

    Args:
        corrected_df (pd.DataFrame): Input data with 'Time'.
        inferred_spikes (dict): Inferred spike trains.

    Returns:
        pd.DataFrame: Time-aligned spikes DataFrame.
    """
    spikes_df = pd.concat([corrected_df['Time'], pd.DataFrame(inferred_spikes)], axis=1)
    return spikes_df

# ----------------------------------
# PEAK PARAMETER CALCULATION
# ----------------------------------
def calculate_peak_parameters(spike_train, time_column, prominence=0.05, distance=5):
    """
    Calculates features for each peak in a spike train.

    Returns:
        List[dict]: Peak properties (index, amplitude, rise/fall time, etc.).
    """
    peaks, properties = find_peaks(spike_train, prominence=prominence, distance=distance)
    troughs = argrelmin(spike_train)[0]
    peak_parameters = []

    for peak_idx in peaks:
        peak_time = time_column[peak_idx]
        peak_value = spike_train[peak_idx]
        preceding = troughs[troughs < peak_idx]
        subsequent = troughs[troughs > peak_idx]
        preceding_idx = preceding.max() if len(preceding) > 0 else None
        subsequent_idx = subsequent.min() if len(subsequent) > 0 else None
        prev_peak = peaks[peaks < peak_idx].max() if len(peaks[peaks < peak_idx]) > 0 else None

        peak_parameters.append({
            'Peak Index': peak_idx,
            'Peak Time': peak_time,
            'Peak Value': peak_value,
            'Delta Amplitude': peak_value - spike_train[preceding_idx] if preceding_idx is not None else np.nan,
            'Rise Time': peak_time - time_column[preceding_idx] if preceding_idx is not None else np.nan,
            'Fall Time': time_column[subsequent_idx] - peak_time if subsequent_idx is not None else np.nan,
            'Oscillation': peak_time - time_column[prev_peak] if prev_peak is not None else np.nan,
            'Confidence of Variance': np.var(spike_train[max(0, peak_idx-5):min(len(spike_train), peak_idx+5)])
        })

    return peak_parameters

def calculate_peaks_for_all_cells(spikes_df, num_cells=None, prominence=0.05, distance=5):
    """
    Calculates peak parameters for each cell in a spike DataFrame.

    Args:
        spikes_df (pd.DataFrame): DataFrame with 'Time' and spike trains.
        num_cells (int): Optional limit to number of cells.
        prominence (float): Peak prominence.
        distance (int): Minimum peak distance.

    Returns:
        pd.DataFrame: DataFrame with all peak parameters across cells.
    """
    all_peaks = []
    cell_columns = spikes_df.columns[1:] if num_cells is None else spikes_df.columns[1:num_cells+1]
    for col in cell_columns:
        print(f"Processing {col}...")
        spike_train = spikes_df[col].values
        time_column = spikes_df['Time'].values
        peak_params = calculate_peak_parameters(spike_train, time_column, prominence, distance)
        for params in peak_params:
            params['Cell'] = col
            all_peaks.append(params)
    return pd.DataFrame(all_peaks)
