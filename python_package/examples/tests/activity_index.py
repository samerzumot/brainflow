import numpy as np
import sys
import os

# add python_package to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from brainflow.data_filter import DataFilter
from brainflow.exit_codes import BrainFlowError, BrainFlowExitCodes


def test_activity_index():
    print("Testing get_activity_index...")

    # 1. Test Constant Signal (AI must be exactly 0.0)
    accel_x = np.full(100, 1.0, dtype=np.float64)
    accel_y = np.full(100, 2.0, dtype=np.float64)
    accel_z = np.full(100, -3.0, dtype=np.float64)

    ai_constant = DataFilter.get_activity_index(accel_x, accel_y, accel_z)
    print(f"Constant signal AI: {ai_constant}")
    assert len(ai_constant) == 1
    assert np.isclose(ai_constant[0], 0.0), f"Expected 0.0, got {ai_constant[0]}"

    # 2. Test Multi-epoch AI calculation
    # Epoch 1: variance = 0 (constant)
    # Epoch 2: variance > 0 (varying)
    N = 50
    x = np.concatenate([np.zeros(N), np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))])
    y = np.concatenate([np.zeros(N), np.cos(np.linspace(0, 2 * np.pi, N, endpoint=False))])
    z = np.concatenate([np.zeros(N), np.zeros(N)])

    ai_epochs = DataFilter.get_activity_index(x, y, z, period=N)
    print(f"Epochs AI: {ai_epochs}")
    assert len(ai_epochs) == 2
    assert np.isclose(ai_epochs[0], 0.0)
    
    # Theoretical variance of sine/cos of amplitude 1 is 0.5
    # var_x = 0.5, var_y = 0.5, var_z = 0 -> total_var = (0.5 + 0.5 + 0)/3 = 1/3
    # AI = sqrt(1/3) = ~0.57735
    expected_ai_epoch1 = np.sqrt((np.var(x[N:]) + np.var(y[N:]) + np.var(z[N:])) / 3.0)
    print(f"Calculated AI: {ai_epochs[1]}, Expected: {expected_ai_epoch1}")
    assert np.isclose(ai_epochs[1], expected_ai_epoch1)

    # 3. Test Invalid Arguments
    try:
        DataFilter.get_activity_index(np.zeros(10), np.zeros(5), np.zeros(10))
        assert False, "Should have raised BrainFlowError for shape mismatch"
    except BrainFlowError as e:
        assert e.exit_code == BrainFlowExitCodes.INVALID_ARGUMENTS_ERROR.value

    print("All activity index tests passed successfully!")


if __name__ == '__main__':
    test_activity_index()
