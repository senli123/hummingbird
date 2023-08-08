#ifndef ENGINE_yolov5Postprocess_wrapper_H
#define ENGINE_yolov5Postprocess_wrapper_H
#include "postprocess_interface.h" 
namespace Engine
{


class Yolov5PostprocessWrapper : public PostprocessInterface{
public:
    virtual bool Init(MapCalcParam& postprocess_input);
    virtual bool Run(float* output);
    virtual std::string GetName()
    {
        return "Yolov5PostprocessWrapper";
    }
    virtual std::vector<InstanceInfo> GetResult();
private:
    std::vector<std::vector<InstanceInfo>> output_infos;
   
};

} // namespace Engine


#endif