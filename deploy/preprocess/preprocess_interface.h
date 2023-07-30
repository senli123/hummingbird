#ifndef ENGINE_preprocess_interface_H
#define ENGINE_preprocess_interface_H
#include <opencv2/opencv.hpp>
#include "../utils/params_define.h"
#include "../utils/register.h"
namespace Engine{
class PreprocessInterface{
public:
    virtual bool Init(MapCalcParam& preprocess_input) = 0;
    virtual bool Run(cv::Mat& input_mat) = 0;
    virtual std::string GetName() = 0;
};
}

#endif