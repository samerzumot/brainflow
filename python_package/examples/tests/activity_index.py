import numpy as np
import sys
import os

# add python_package to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from brainflow.data_filter import DataFilter
from brainflow.exit_codes import BrainFlowError, BrainFlowExitCodes


def test_activity_index():
    print("Testing get_activity_index...")
    fs = 50

    # 1. Test Constant Signal (AI must be exactly 0.0)
    accel_x = np.full(100, 1.0, dtype=np.float64)
    accel_y = np.full(100, 2.0, dtype=np.float64)
    accel_z = np.full(100, -3.0, dtype=np.float64)

    ai_constant = DataFilter.get_activity_index(accel_x, accel_y, accel_z, sampling_rate=fs)
    print(f"Constant signal AI: {ai_constant}")
    assert len(ai_constant) == 1
    assert np.isclose(ai_constant[0], 0.0), f"Expected 0.0, got {ai_constant[0]}"

    # 2. Test Stationary Noisy Signal with Baseline Noise Cancellation
    # Stationary sensor produces noise variance. With baseline noise subtracted, AI should be 0.0.
    np.random.seed(42)
    noise_x = np.random.normal(0, 0.2, 200)  # theoretical var ~ 0.04
    noise_y = np.random.normal(0, 0.2, 200)
    noise_z = np.random.normal(0, 0.2, 200)

    sample_var_x = float(np.var(noise_x[:fs]))
    sample_var_y = float(np.var(noise_y[:fs]))
    sample_var_z = float(np.var(noise_z[:fs]))

    # If we pass the exact baseline noise variance of rest data, AI must be 0.0
    ai_rest = DataFilter.get_activity_index(
        noise_x[:fs], noise_y[:fs], noise_z[:fs],
        sampling_rate=fs, period=fs,
        noise_var_x=sample_var_x, noise_var_y=sample_var_y, noise_var_z=sample_var_z
    )
    print(f"Stationary noise cancelled AI: {ai_rest}")
    assert np.isclose(ai_rest[0], 0.0)

    # If baseline noise exceeds signal variance, clamping to 0 must prevent negative values
    ai_clamped = DataFilter.get_activity_index(
        noise_x[:fs], noise_y[:fs], noise_z[:fs],
        sampling_rate=fs, period=fs,
        noise_var_x=1.0, noise_var_y=1.0, noise_var_z=1.0
    )
    assert np.isclose(ai_clamped[0], 0.0)

    # 3. Test Signal with Known Variance minus Noise Variance
    # Sine wave with amplitude 1.0 has variance = 0.5
    t = np.linspace(0, 1.0, fs, endpoint=False)
    sig_x = np.sin(2 * np.pi * 2 * t)  # var = 0.5
    sig_y = np.cos(2 * np.pi * 2 * t)  # var = 0.5
    sig_z = np.zeros(fs)               # var = 0.0
    noise_val = 0.1

    actual_var_x = float(np.var(sig_x))
    actual_var_y = float(np.var(sig_y))
    actual_var_z = float(np.var(sig_z))
    expected_ai = np.sqrt(max(0.0, ((actual_var_x - noise_val) + (actual_var_y - noise_val) + (actual_var_z - 0.0)) / 3.0))

    ai_test = DataFilter.get_activity_index(
        sig_x, sig_y, sig_z, sampling_rate=fs, period=fs,
        noise_var_x=noise_val, noise_var_y=noise_val, noise_var_z=0.0
    )
    print(f"Known signal AI: {ai_test[0]}, Expected: {expected_ai}")
    assert np.isclose(ai_test[0], expected_ai)

    # 4. Test Epoch Semantics (Bai et al. 2016 summing adjacent 1-second AIs)
    # Second 1: low motion wave (amplitude 0.5, var = 0.125)
    # Second 2: high motion wave (amplitude 2.0, var = 2.0)
    t1 = np.linspace(0, 1.0, fs, endpoint=False)
    s1_x = 0.5 * np.sin(2 * np.pi * 2 * t1)
    s1_y = 0.5 * np.cos(2 * np.pi * 2 * t1)
    s1_z = np.zeros(fs)

    t2 = np.linspace(0, 1.0, fs, endpoint=False)
    s2_x = 2.0 * np.sin(2 * np.pi * 2 * t2)
    s2_y = 2.0 * np.cos(2 * np.pi * 2 * t2)
    s2_z = np.zeros(fs)

    # 1-second AIs individually
    ai_sec1 = DataFilter.get_activity_index(s1_x, s1_y, s1_z, sampling_rate=fs, period=fs)[0]
    ai_sec2 = DataFilter.get_activity_index(s2_x, s2_y, s2_z, sampling_rate=fs, period=fs)[0]

    # Combined 2-second signal
    comb_x = np.concatenate([s1_x, s2_x])
    comb_y = np.concatenate([s1_y, s2_y])
    comb_z = np.concatenate([s1_z, s2_z])

    # 2-second epoch AI should be EXACTLY ai_sec1 + ai_sec2 per Bai et al. 2016
    ai_2sec = DataFilter.get_activity_index(comb_x, comb_y, comb_z, sampling_rate=fs, period=2 * fs)
    assert len(ai_2sec) == 1
    print(f"2-second epoch AI: {ai_2sec[0]}, Sum of 1s AIs: {ai_sec1 + ai_sec2}")
    assert np.isclose(ai_2sec[0], ai_sec1 + ai_sec2)

    # Show that it is NOT simply the variance recomputed over the entire 2-second period
    recomputed_full_var_ai = np.sqrt((np.var(comb_x) + np.var(comb_y) + np.var(comb_z)) / 3.0)
    print(f"Recomputed full variance AI: {recomputed_full_var_ai}")
    assert not np.isclose(ai_2sec[0], recomputed_full_var_ai)

    # 5. Test Multi-epoch AI calculation (4 seconds -> two 2-second epochs)
    comb4_x = np.concatenate([comb_x, comb_x])
    comb4_y = np.concatenate([comb_y, comb_y])
    comb4_z = np.concatenate([comb_z, comb_z])
    ai_4sec = DataFilter.get_activity_index(comb4_x, comb4_y, comb4_z, sampling_rate=fs, period=2 * fs)
    assert len(ai_4sec) == 2
    assert np.isclose(ai_4sec[0], ai_2sec[0])
    assert np.isclose(ai_4sec[1], ai_2sec[0])

    # 6. Test Invalid Arguments
    # Shape mismatch
    try:
        DataFilter.get_activity_index(np.zeros(10), np.zeros(5), np.zeros(10), sampling_rate=fs)
        assert False, "Should have raised BrainFlowError for shape mismatch"
    except BrainFlowError as e:
        assert e.exit_code == BrainFlowExitCodes.INVALID_ARGUMENTS_ERROR.value

    # Empty inputs
    try:
        DataFilter.get_activity_index(np.array([]), np.array([]), np.array([]), sampling_rate=fs)
        assert False, "Should have raised BrainFlowError for empty inputs"
    except BrainFlowError as e:
        assert e.exit_code == BrainFlowExitCodes.INVALID_ARGUMENTS_ERROR.value

    # Invalid sampling rate
    try:
        DataFilter.get_activity_index(np.zeros(100), np.zeros(100), np.zeros(100), sampling_rate=0)
        assert False, "Should have raised BrainFlowError for zero sampling rate"
    except BrainFlowError as e:
        assert e.exit_code == BrainFlowExitCodes.INVALID_ARGUMENTS_ERROR.value

    # Period greater than data length
    try:
        DataFilter.get_activity_index(np.zeros(100), np.zeros(100), np.zeros(100), sampling_rate=fs, period=200)
        assert False, "Should have raised BrainFlowError for period > data_len"
    except BrainFlowError as e:
        assert e.exit_code == BrainFlowExitCodes.INVALID_ARGUMENTS_ERROR.value

    # Period not an integer multiple of sampling rate
    try:
        DataFilter.get_activity_index(np.zeros(100), np.zeros(100), np.zeros(100), sampling_rate=fs, period=75)
        assert False, "Should have raised BrainFlowError for period not multiple of fs"
    except BrainFlowError as e:
        assert e.exit_code == BrainFlowExitCodes.INVALID_ARGUMENTS_ERROR.value

    # Negative noise variance
    try:
        DataFilter.get_activity_index(np.zeros(100), np.zeros(100), np.zeros(100), sampling_rate=fs, noise_var_x=-0.5)
        assert False, "Should have raised BrainFlowError for negative noise variance"
    except BrainFlowError as e:
        assert e.exit_code == BrainFlowExitCodes.INVALID_ARGUMENTS_ERROR.value

    print("All activity index tests passed successfully!")


if __name__ == '__main__':
    test_activity_index()
