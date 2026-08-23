#pragma once

#include <deque>
#include <memory>

#include "base_classifier.h"
#include "brainflow_constants.h"
#include "brainflow_model_params.h"


class MovingAverageClassifier : public BaseClassifier
{
protected:
    int window_len;
    std::deque<double> buffer;
    double sum;
    std::shared_ptr<BaseClassifier> base_classifier;

    int parse_window_len ();

public:
    MovingAverageClassifier (struct BrainFlowModelParams params);
    ~MovingAverageClassifier ();

    int prepare () override;
    int predict (double *data, int data_len, double *output, int *output_len) override;
    int release () override;
};
