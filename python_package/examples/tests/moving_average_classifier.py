import numpy as np
import sys
import os

# add python_package to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
from brainflow.exit_codes import BrainFlowError, BrainFlowExitCodes


def test_moving_average_classifier():
    print("Testing MovingAverageClassifier...")

    # 1. Test USER_DEFINED stream with window_len = 3
    params = BrainFlowModelParams(
        BrainFlowMetrics.USER_DEFINED.value,
        BrainFlowClassifiers.MOVING_AVERAGE_CLASSIFIER.value
    )
    params.other_info = "3"

    model = MLModel(params)
    model.prepare()

    # Step 1: Input 10.0 -> Avg: 10.0
    out1 = model.predict(np.array([10.0], dtype=np.float64))
    print(f"Step 1: In=10.0, Out={out1[0]}")
    assert np.isclose(out1[0], 10.0)

    # Step 2: Input 20.0 -> Avg: (10 + 20) / 2 = 15.0
    out2 = model.predict(np.array([20.0], dtype=np.float64))
    print(f"Step 2: In=20.0, Out={out2[0]}")
    assert np.isclose(out2[0], 15.0)

    # Step 3: Input 30.0 -> Avg: (10 + 20 + 30) / 3 = 20.0
    out3 = model.predict(np.array([30.0], dtype=np.float64))
    print(f"Step 3: In=30.0, Out={out3[0]}")
    assert np.isclose(out3[0], 20.0)

    # Step 4: Input 40.0 -> Avg: (20 + 30 + 40) / 3 = 30.0 (oldest 10.0 dropped)
    out4 = model.predict(np.array([40.0], dtype=np.float64))
    print(f"Step 4: In=40.0, Out={out4[0]}")
    assert np.isclose(out4[0], 30.0)

    model.release()

    # 2. Test MINDFULNESS metric with MOVING_AVERAGE_CLASSIFIER
    mf_params = BrainFlowModelParams(
        BrainFlowMetrics.MINDFULNESS.value,
        BrainFlowClassifiers.MOVING_AVERAGE_CLASSIFIER.value
    )
    mf_params.other_info = '{"window_len": 4}'

    mf_model = MLModel(mf_params)
    mf_model.prepare()

    # 5 band powers input
    feature_vector = np.array([0.1, 0.2, 0.3, 0.2, 0.2], dtype=np.float64)
    mf_out1 = mf_model.predict(feature_vector)
    print(f"Mindfulness moving avg 1: {mf_out1[0]}")
    assert 0.0 <= mf_out1[0] <= 1.0

    mf_out2 = mf_model.predict(feature_vector)
    print(f"Mindfulness moving avg 2: {mf_out2[0]}")
    assert np.isclose(mf_out1[0], mf_out2[0])

    mf_model.release()

    print("All MovingAverageClassifier tests passed successfully!")


if __name__ == '__main__':
    test_moving_average_classifier()
