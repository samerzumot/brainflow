#pragma once

#include <string>

#include "base_classifier.h"
#include "runtime_dll_loader.h"


// DynLibClassifier loads user-provided shared libraries (.so / .dll / .dylib) exporting
// "prepare", "predict", and "release" C functions.
// Note for plugin authors: BaseClassifier inspects params.other_info for moving average
// configuration ("moving_average", "window_len", "period", or bare positive integer / "true").
// If enabled, moving average smoothing is applied to the plugin's output. To avoid unintended
// smoothing, custom plugins using other_info should use unique JSON or key-value keys.
class DynLibClassifier : public BaseClassifier
{
public:
    DynLibClassifier (struct BrainFlowModelParams params) : BaseClassifier (params)
    {
        dll_loader = NULL;
    }

    ~DynLibClassifier () override
    {
        skip_logs = true;
        release ();
    }

protected:
    int prepare_classifier () override;
    int calculate (double *data, int data_len, double *output, int *output_len) override;
    int release_classifier () override;

    virtual std::string get_dyn_lib_path ()
    {
        return params.file;
    }

    DLLLoader *dll_loader;
};
