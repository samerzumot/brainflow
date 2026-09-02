import numpy as np
import sys
import os

# add python_package to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
from brainflow.exit_codes import BrainFlowError, BrainFlowExitCodes


def test_moving_average_classifier():
    print("Testing moving average option on ML classifiers...")

    v1 = np.array([0.1, 0.2, 0.3, 0.2, 0.2], dtype=np.float64)
    v2 = np.array([0.5, 0.1, 0.1, 0.1, 0.2], dtype=np.float64)

    # 1. Baseline: MINDFULNESS + DEFAULT_CLASSIFIER without moving average
    raw_params = BrainFlowModelParams(
        BrainFlowMetrics.MINDFULNESS.value,
        BrainFlowClassifiers.DEFAULT_CLASSIFIER.value
    )
    raw_model = MLModel(raw_params)
    raw_model.prepare()
    raw_score_1 = raw_model.predict(v1)[0]
    raw_score_2 = raw_model.predict(v2)[0]
    raw_model.release()

    print(f"Raw scores: v1={raw_score_1:.6f}, v2={raw_score_2:.6f}")
    assert raw_score_1 != raw_score_2

    # 2. Test JSON config with explicit window_len: '{"window_len": 3}'
    mf_params = BrainFlowModelParams(
        BrainFlowMetrics.MINDFULNESS.value,
        BrainFlowClassifiers.DEFAULT_CLASSIFIER.value
    )
    mf_params.other_info = '{"window_len": 3}'
    mf_model = MLModel(mf_params)
    mf_model.prepare()

    # Step 1: In=v1 -> out1 = raw_score_1
    out1 = mf_model.predict(v1)[0]
    print(f"Step 1 (v1): out={out1:.6f}, expected={raw_score_1:.6f}")
    assert np.isclose(out1, raw_score_1)

    # Step 2: In=v2 -> out2 = (raw1 + raw2) / 2
    out2 = mf_model.predict(v2)[0]
    expected_2 = (raw_score_1 + raw_score_2) / 2.0
    print(f"Step 2 (v2): out={out2:.6f}, expected={expected_2:.6f}")
    assert np.isclose(out2, expected_2)

    # Step 3: In=v2 -> out3 = (raw1 + raw2 + raw2) / 3
    out3 = mf_model.predict(v2)[0]
    expected_3 = (raw_score_1 + 2.0 * raw_score_2) / 3.0
    print(f"Step 3 (v2): out={out3:.6f}, expected={expected_3:.6f}")
    assert np.isclose(out3, expected_3)

    # Step 4: In=v2 -> out4 = (raw2 + raw2 + raw2) / 3 = raw2 (oldest raw1 popped!)
    out4 = mf_model.predict(v2)[0]
    expected_4 = raw_score_2
    print(f"Step 4 (v2): out={out4:.6f}, expected={expected_4:.6f}")
    assert np.isclose(out4, expected_4)

    mf_model.release()

    # 3. Test RESTFULNESS metric with moving average: '{"moving_average": true, "window_len": 2}'
    rf_params = BrainFlowModelParams(
        BrainFlowMetrics.RESTFULNESS.value,
        BrainFlowClassifiers.DEFAULT_CLASSIFIER.value
    )
    rf_params.other_info = '{"moving_average": true, "window_len": 2}'
    rf_model = MLModel(rf_params)
    rf_model.prepare()

    raw_rf_1 = 1.0 - raw_score_1
    raw_rf_2 = 1.0 - raw_score_2

    rf_out1 = rf_model.predict(v1)[0]
    assert np.isclose(rf_out1, raw_rf_1)

    rf_out2 = rf_model.predict(v2)[0]
    assert np.isclose(rf_out2, (raw_rf_1 + raw_rf_2) / 2.0)

    rf_model.release()

    # 4. Test default window fallback: '{"moving_average": true}' (default window_len = 5)
    def_params = BrainFlowModelParams(
        BrainFlowMetrics.MINDFULNESS.value,
        BrainFlowClassifiers.DEFAULT_CLASSIFIER.value
    )
    def_params.other_info = '{"moving_average": true}'
    def_model = MLModel(def_params)
    def_model.prepare()

    # Feed 5 identical samples, then a 6th different sample
    for _ in range(5):
        def_model.predict(v1)
    # The 6th prediction should be (4 * raw1 + 1 * raw2) / 5
    def_out6 = def_model.predict(v2)[0]
    assert np.isclose(def_out6, (4.0 * raw_score_1 + raw_score_2) / 5.0)

    def_model.release()

    # 5. Test key-value string format: 'moving_average=3'
    kv_params = BrainFlowModelParams(
        BrainFlowMetrics.MINDFULNESS.value,
        BrainFlowClassifiers.DEFAULT_CLASSIFIER.value
    )
    kv_params.other_info = "moving_average=3"
    kv_model = MLModel(kv_params)
    kv_model.prepare()

    kv_out1 = kv_model.predict(v1)[0]
    assert np.isclose(kv_out1, raw_score_1)
    kv_out2 = kv_model.predict(v2)[0]
    assert np.isclose(kv_out2, (raw_score_1 + raw_score_2) / 2.0)
    kv_model.release()

    # 6. Test integer string format: '3'
    num_params = BrainFlowModelParams(
        BrainFlowMetrics.MINDFULNESS.value,
        BrainFlowClassifiers.DEFAULT_CLASSIFIER.value
    )
    num_params.other_info = "3"
    num_model = MLModel(num_params)
    num_model.prepare()

    num_out1 = num_model.predict(v1)[0]
    assert np.isclose(num_out1, raw_score_1)
    num_out2 = num_model.predict(v2)[0]
    assert np.isclose(num_out2, (raw_score_1 + raw_score_2) / 2.0)
    num_model.release()

    # 7. Negative test: Unrelated other_info string does NOT activate moving average
    neg_params = BrainFlowModelParams(
        BrainFlowMetrics.MINDFULNESS.value,
        BrainFlowClassifiers.DEFAULT_CLASSIFIER.value
    )
    neg_params.other_info = '{"unrelated_key": "some_value"}'
    neg_model = MLModel(neg_params)
    neg_model.prepare()

    neg_out1 = neg_model.predict(v1)[0]
    assert np.isclose(neg_out1, raw_score_1)
    neg_out2 = neg_model.predict(v2)[0]
    # Since moving average is NOT enabled, out2 should be raw_score_2, NOT an average
    assert np.isclose(neg_out2, raw_score_2)
    neg_model.release()

    print("All moving average classifier tests passed successfully!")


if __name__ == '__main__':
    test_moving_average_classifier()
