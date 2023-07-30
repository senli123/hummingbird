#ifndef ENGINE_reszie_wrapper_H
#define ENGINE_resize_wrapper_H
#include "preprocess_interface.h" 
namespace Engine
{
class NormalizeWrapper : public PreprocessInterface{
public:
    virtual bool Init(MapCalcParam& preprocess_input);
    virtual bool Run(cv::Mat& input_mat);
    virtual std::string GetName()
    {
        return "NormalizeWrapper";
    }

private:
    
    bool _limit = false;
    bool _normalize = false;
    cv::Scalar _mean = cv::Scalar(0,0,0);
    cv::Scalar _std = cv::Scalar(0,0,0);
};

} // namespace Engine


#endif