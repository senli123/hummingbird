    #ifndef ENGINE_yolov5Preprocess_wrapper_H
#define ENGINE_yolov5Preprocess_wrapper_H
#include "preprocess_interface.h" 
namespace Engine
{
class Yolov5PreprocessWrapper : public PreprocessInterface{
public:
    virtual bool Init(MapCalcParam& preprocess_input);
    virtual bool Run(cv::Mat& input_mat);
    virtual std::string GetName()
    {
        return " Yolov5Preprocess";
    }

private:
   
    
};

} // namespace Engine


#endif