#ifndef ENGINE_reszie_wrapper_H
#define ENGINE_resize_wrapper_H
#include "preprocess_interface.h" 
namespace Engine
{
class ResizeWrapper : public PreprocessInterface{
public:
    virtual bool Init(MapCalcParam& preprocess_input);
    virtual bool Run(cv::Mat& input_mat);
    virtual std::string GetName()
    {
        return "ResizeWrapper";
    }

private:
    bool _bgr2rgb = false;  
    bool _keep_ratio = true;
    cv::Size _size = cv::Size(0,0);
};

} // namespace Engine


#endif