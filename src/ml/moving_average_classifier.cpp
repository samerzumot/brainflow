#include <cmath>
#include <cstdlib>
#include <string>

#include "brainflow_constants.h"
#include "json.hpp"
#include "mindfulness_classifier.h"
#include "moving_average_classifier.h"
#include "restfulness_classifier.h"

using json = nlohmann::json;


MovingAverageClassifier::MovingAverageClassifier (struct BrainFlowModelParams model_params)
    : BaseClassifier (model_params)
{
    window_len = 5;
    sum = 0.0;
    base_classifier = NULL;

    if (params.metric == (int)BrainFlowMetrics::MINDFULNESS)
    {
        base_classifier = std::shared_ptr<BaseClassifier> (new MindfulnessClassifier (params));
    }
    else if (params.metric == (int)BrainFlowMetrics::RESTFULNESS)
    {
        base_classifier = std::shared_ptr<BaseClassifier> (new RestfulnessClassifier (params));
    }
}

MovingAverageClassifier::~MovingAverageClassifier ()
{
    buffer.clear ();
    sum = 0.0;
    base_classifier = NULL;
}

int MovingAverageClassifier::parse_window_len ()
{
    int len = 5;
    if (!params.other_info.empty ())
    {
        try
        {
            if (params.other_info.find ("{") != std::string::npos)
            {
                json j = json::parse (params.other_info);
                if (j.contains ("window_len"))
                {
                    len = j["window_len"].get<int> ();
                }
                else if (j.contains ("period"))
                {
                    len = j["period"].get<int> ();
                }
            }
            else
            {
                len = std::stoi (params.other_info);
            }
        }
        catch (...)
        {
            safe_logger (spdlog::level::warn,
                "Unable to parse window_len from other_info: {}. Using default value of 5.",
                params.other_info);
            len = 5;
        }
    }
    if (len <= 0)
    {
        len = 5;
    }
    return len;
}

int MovingAverageClassifier::prepare ()
{
    buffer.clear ();
    sum = 0.0;
    window_len = parse_window_len ();

    if (base_classifier != NULL)
    {
        return base_classifier->prepare ();
    }
    return (int)BrainFlowExitCodes::STATUS_OK;
}

int MovingAverageClassifier::predict (
    double *data, int data_len, double *output, int *output_len)
{
    if ((data == NULL) || (output == NULL) || (data_len <= 0))
    {
        safe_logger (spdlog::level::err, "Incorrect arguments for predict.");
        return (int)BrainFlowExitCodes::INVALID_ARGUMENTS_ERROR;
    }

    double raw_score = 0.0;
    if (base_classifier != NULL)
    {
        double base_output = 0.0;
        int base_output_len = 0;
        int res = base_classifier->predict (data, data_len, &base_output, &base_output_len);
        if (res != (int)BrainFlowExitCodes::STATUS_OK)
        {
            return res;
        }
        raw_score = base_output;
    }
    else
    {
        raw_score = data[0];
    }

    buffer.push_back (raw_score);
    sum += raw_score;
    if ((int)buffer.size () > window_len)
    {
        sum -= buffer.front ();
        buffer.pop_front ();
    }

    *output = sum / buffer.size ();
    *output_len = 1;
    return (int)BrainFlowExitCodes::STATUS_OK;
}

int MovingAverageClassifier::release ()
{
    buffer.clear ();
    sum = 0.0;
    if (base_classifier != NULL)
    {
        return base_classifier->release ();
    }
    return (int)BrainFlowExitCodes::STATUS_OK;
}
