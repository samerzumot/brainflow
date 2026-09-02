#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "base_classifier.h"
#include "brainflow_constants.h"
#include "json.hpp"
#include "spdlog/sinks/null_sink.h"

using json = nlohmann::json;

#define LOGGER_NAME "ml_logger"

#ifdef __ANDROID__
#include "spdlog/sinks/android_sink.h"
std::shared_ptr<spdlog::logger> BaseClassifier::ml_logger =
    spdlog::android_logger (LOGGER_NAME, "ml_ndk_logger");
#else
std::shared_ptr<spdlog::logger> BaseClassifier::ml_logger = spdlog::stderr_logger_mt (LOGGER_NAME);
#endif

int BaseClassifier::set_log_level (int level)
{
    int log_level = level;
    if (level > 6)
    {
        log_level = 6;
    }
    if (level < 0)
    {
        log_level = 0;
    }
    try
    {
        BaseClassifier::ml_logger->set_level (spdlog::level::level_enum (log_level));
        BaseClassifier::ml_logger->flush_on (spdlog::level::level_enum (log_level));
    }
    catch (...)
    {
        return (int)BrainFlowExitCodes::GENERAL_ERROR;
    }
    return (int)BrainFlowExitCodes::STATUS_OK;
}

int BaseClassifier::set_log_file (const char *log_file)
{
#ifdef __ANDROID__
    BaseClassifier::ml_logger->error ("For Android set_log_file is unavailable");
    return (int)BrainFlowExitCodes::GENERAL_ERROR;
#else
    try
    {
        spdlog::level::level_enum level = BaseClassifier::ml_logger->level ();
        BaseClassifier::ml_logger = spdlog::create<spdlog::sinks::null_sink_st> (
            "null_logger"); // to dont set logger to nullptr and avoid race condition
        spdlog::drop (LOGGER_NAME);
        BaseClassifier::ml_logger = spdlog::basic_logger_mt (LOGGER_NAME, log_file);
        BaseClassifier::ml_logger->set_level (level);
        BaseClassifier::ml_logger->flush_on (level);
        spdlog::drop ("null_logger");
    }
    catch (...)
    {
        return (int)BrainFlowExitCodes::GENERAL_ERROR;
    }
    return (int)BrainFlowExitCodes::STATUS_OK;
#endif
}

void BaseClassifier::parse_moving_average_params ()
{
    use_moving_average = false;
    moving_average_window = 0;

    if (params.other_info.empty ())
    {
        return;
    }

    std::string info = params.other_info;
    try
    {
        if (info.find ('{') != std::string::npos)
        {
            json j = json::parse (info);
            if (j.contains ("moving_average"))
            {
                if (j["moving_average"].is_boolean ())
                {
                    use_moving_average = j["moving_average"].get<bool> ();
                    if (use_moving_average)
                    {
                        moving_average_window = DEFAULT_MOVING_AVERAGE_WINDOW;
                    }
                }
                else if (j["moving_average"].is_number_integer ())
                {
                    moving_average_window = j["moving_average"].get<int> ();
                    use_moving_average = (moving_average_window > 0);
                }
            }
            if (j.contains ("window_len"))
            {
                moving_average_window = j["window_len"].get<int> ();
                use_moving_average = true;
            }
            else if (j.contains ("period"))
            {
                moving_average_window = j["period"].get<int> ();
                use_moving_average = true;
            }
        }
        else if (info.find ('=') != std::string::npos)
        {
            std::stringstream ss (info);
            std::string token;
            while (std::getline (ss, token, ';'))
            {
                size_t eq = token.find ('=');
                if (eq != std::string::npos)
                {
                    std::string k = token.substr (0, eq);
                    std::string v = token.substr (eq + 1);
                    k.erase (0, k.find_first_not_of (" \t\r\n"));
                    k.erase (k.find_last_not_of (" \t\r\n") + 1);
                    v.erase (0, v.find_first_not_of (" \t\r\n"));
                    v.erase (v.find_last_not_of (" \t\r\n") + 1);
                    std::transform (k.begin (), k.end (), k.begin (), ::tolower);
                    std::transform (v.begin (), v.end (), v.begin (), ::tolower);

                    if (k == "moving_average")
                    {
                        if (v == "true" || v == "1")
                        {
                            use_moving_average = true;
                            if (moving_average_window <= 0)
                            {
                                moving_average_window = DEFAULT_MOVING_AVERAGE_WINDOW;
                            }
                        }
                        else if (v == "false" || v == "0")
                        {
                            use_moving_average = false;
                        }
                        else
                        {
                            try
                            {
                                moving_average_window = std::stoi (v);
                                use_moving_average = (moving_average_window > 0);
                            }
                            catch (...)
                            {
                            }
                        }
                    }
                    else if (k == "window_len" || k == "period")
                    {
                        try
                        {
                            moving_average_window = std::stoi (v);
                            use_moving_average = true;
                        }
                        catch (...)
                        {
                        }
                    }
                }
            }
        }
        else
        {
            try
            {
                size_t idx = 0;
                int val = std::stoi (info, &idx);
                if ((idx == info.length ()) && (val > 0))
                {
                    moving_average_window = val;
                    use_moving_average = true;
                }
            }
            catch (...)
            {
                std::string lower_info = info;
                lower_info.erase (0, lower_info.find_first_not_of (" \t\r\n"));
                lower_info.erase (lower_info.find_last_not_of (" \t\r\n") + 1);
                std::transform (
                    lower_info.begin (), lower_info.end (), lower_info.begin (), ::tolower);
                if ((lower_info == "moving_average") || (lower_info == "true"))
                {
                    use_moving_average = true;
                    moving_average_window = DEFAULT_MOVING_AVERAGE_WINDOW;
                }
            }
        }
    }
    catch (std::exception &e)
    {
        safe_logger (spdlog::level::warn,
            "Failed to parse moving average from other_info: {}. Moving average disabled.",
            e.what ());
        use_moving_average = false;
        moving_average_window = 0;
    }

    if (use_moving_average)
    {
        if (moving_average_window <= 0)
        {
            safe_logger (spdlog::level::warn,
                "Invalid moving average window ({}), using default {}.", moving_average_window,
                DEFAULT_MOVING_AVERAGE_WINDOW);
            moving_average_window = DEFAULT_MOVING_AVERAGE_WINDOW;
        }
    }
}

void BaseClassifier::reset_moving_average ()
{
    window_data.clear ();
}

void BaseClassifier::apply_moving_average (double *output, int *output_len)
{
    if ((output == NULL) || (output_len == NULL) || (*output_len <= 0))
    {
        return;
    }

    if (window_data.empty () || ((int)window_data.front ().size () != *output_len))
    {
        window_data.clear ();
    }

    window_data.push_back (std::vector<double> (output, output + *output_len));
    while ((int)window_data.size () > moving_average_window)
    {
        window_data.pop_front ();
    }

    for (int i = 0; i < *output_len; i++)
    {
        double sum = 0.0;
        for (size_t w = 0; w < window_data.size (); w++)
        {
            sum += window_data[w][i];
        }
        output[i] = sum / window_data.size ();
    }
}

int BaseClassifier::prepare ()
{
    parse_moving_average_params ();
    reset_moving_average ();
    return prepare_classifier ();
}

int BaseClassifier::predict (double *data, int data_len, double *output, int *output_len)
{
    int res = calculate (data, data_len, output, output_len);
    if (res != (int)BrainFlowExitCodes::STATUS_OK)
    {
        return res;
    }
    if (use_moving_average)
    {
        apply_moving_average (output, output_len);
    }
    return (int)BrainFlowExitCodes::STATUS_OK;
}

int BaseClassifier::release ()
{
    reset_moving_average ();
    return release_classifier ();
}
