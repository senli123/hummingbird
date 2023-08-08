#ifndef ENGINE_postprocess_interface_H
#define ENGINE_postprocess_interface_H
#include <opencv2/opencv.hpp>
#include "../utils/params_define.h"
#include "../utils/register.h"
namespace Engine{
class PostprocessInterface{
public:
    virtual bool Init(MapCalcParam& postprocess_input) = 0;
    virtual bool Run(float* output) = 0;
    virtual std::string GetName() = 0;
    virtual std::vector<InstanceInfo> GetResult() = 0;
};
}

#endif