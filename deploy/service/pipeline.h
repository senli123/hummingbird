#ifndef Engine_pipeline_h
#define Engine_pipeline_h
#include <json/json.h>
#include "create_interface.h"
#include <opencv2/opencv.hpp>
#include "visual_tools.h"
namespace Engine
{
class Pipeline{
public:
    bool CreatePipeline(std::string& json_path);
    bool RunPipeline(std::string& image_path);
    bool DestroyPipeline();
private:
    bool PhaseJson(std::string& json_path, 
        std::vector<std::pair<std::string, MapCalcParam>>& preprocess_info,
        std::vector<std::pair<std::string, MapCalcParam>>& backend_info,
        std::vector<std::pair<std::string, MapCalcParam>>& postprocess_info,
        std::string& visual_type);
    bool PhaseParams(Json::Value &infer_info, std::vector<std::pair<std::string, MapCalcParam>>& info_list);
private:
    std::vector<PreprocessInterface*> preprocess_pipeline;
    std::vector<BackendInterface*> backend_model;
    std::vector<PostprocessInterface*> postprocess_pipeline;
    std::string visual_type;
    float* input_buffer = nullptr;
    float* output_buffer = nullptr;
    int input_h;
    int input_w;
};   
} // namespace Engine

#endif