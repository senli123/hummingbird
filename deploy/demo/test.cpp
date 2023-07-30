#include <iostream>
#include "../service/pipeline.h"

int main()
{
    std::string json_path = "/workspace/lisen/tensorrt/my_tensorrt/config/example.json";
    std::string image_path = "";
    Engine::Pipeline test_pipeline;
    if (!test_pipeline.CreatePipeline(json_path))
    {
        return false;
    }
    if (!test_pipeline.RunPipeline(image_path))
    {
        return false;
    }
    if (!test_pipeline.DestroyPipeline())
    {
        return false;
    }
    return true;
}