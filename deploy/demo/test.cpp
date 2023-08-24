#include <iostream>
#include "../service/pipeline.h"

int main()
{

     std::string json_path = "/workspace/lisen/tensorrt/my_tensorrt/config/deploy/squeezenet_ncnn.json";
    std::string image_path = "/workspace/lisen/tensorrt/my_tensorrt/img/dog.jpg";
    Engine::Pipeline test_pipeline;
    if (!test_pipeline.CreatePipeline(json_path))
    {
        return false;
    }
    for(int i = 0; i < 10; i++){
        if (!test_pipeline.RunPipeline(image_path))
        {
            return false;
        }
    }
   
    if (!test_pipeline.DestroyPipeline())
    {
        return false;
    }
    return true;
}