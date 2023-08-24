#ifndef ENGINE_yolov5OrtPostprocess_wrapper_H
#define ENGINE_yolov5OrtPostprocess_wrapper_H
#include "postprocess_interface.h" 
namespace Engine
{


class Yolov5OrtPostprocessWrapper : public PostprocessInterface{
public:
    virtual bool Init(MapCalcParam& postprocess_input);
    virtual bool Run(float* output);
    virtual std::string GetName()
    {
        return "Yolov5OrtPostprocessWrapper";
    }
    virtual std::vector<InstanceInfo> GetResult();
    void getBestClassInfo(float* it, const int& numClasses,
                                    float& bestConf, int& bestClassId);

    void scaleCoords(const cv::Size& imageShape, cv::Rect& coords, const cv::Size& imageOriginalShape);
private:
    std::vector<std::vector<InstanceInfo>> output_infos;
   
};

} // namespace Engine


#endif