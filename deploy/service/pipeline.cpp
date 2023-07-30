#include "pipeline.h"

namespace Engine
{
bool Pipeline::CreatePipeline(std::string& json_path)
{
    //先清空已经注册的pipeline
    this->DestroyPipeline();
    std::vector<std::pair<std::string, MapCalcParam>> preprocess_info;
    std::vector<std::pair<std::string, MapCalcParam>> backend_info;
    std::vector<std::pair<std::string, MapCalcParam>> postprocess_info;
    if(!this->PhaseJson(json_path,preprocess_info,backend_info,postprocess_info)){
        std::cerr << "PhaseJson failed " << '\n';
        return false;
    }
    //依次去注册
    for (auto single_preprocess_info : preprocess_info) {
        std::string preprocess_type = single_preprocess_info.first;
        MapCalcParam preprocess_params = single_preprocess_info.second;
        PreprocessInterface* preprocess_node = PreprocessFactory::createPreprocessInterface(preprocess_type);
        if (preprocess_node == nullptr)
        {
            std::cerr << "preprocess_node: "<< preprocess_type << "not in" << '\n';
            Store::getInstance()->getPreprocessKeys();
            return false;
        }
        if(!preprocess_node->Init(preprocess_params))
        {
            std::cerr << "preprocess_node: "<< preprocess_type<< "init failed" << '\n';
            return false;
        }
        this->preprocess_pipeline.push_back(preprocess_node);
    }

    for (auto single_backend_info : backend_info) {
        std::string backend_type = single_backend_info.first;
        MapCalcParam backend_params = single_backend_info.second;
        BackendInterface* backend_node = BackendFactory::createBackendInterface(backend_type);
        if (backend_node == nullptr)
        {
            std::cerr << "backend_node: "<< backend_node << "not in" << '\n';
            Store::getInstance()->getBackendKeys();
            return false;
        }
        if(!backend_node->Init(backend_params))
        {
            std::cerr << "backend_node: "<< backend_type << "init failed" << '\n';
            return false;
        }
        this->backend_model.push_back(backend_node);
    }

    for (auto single_postprocess_info : postprocess_info) {
        std::string postprocess_type = single_postprocess_info.first;
        MapCalcParam postprocess_params = single_postprocess_info.second;
        PostprocessInterface* postprocess_node = PostprocessFactory::createPostprocessInterface(postprocess_type);
        if (postprocess_node == nullptr)
        {
            std::cerr << "postprocess_node: "<< postprocess_type << "not in" << '\n';
            Store::getInstance()->getPostprocessKeys();
            return false;
        }
        if(!postprocess_node->Init(postprocess_params))
        {
            std::cerr << "postprocess_node: "<< postprocess_type<< "init failed" << '\n';
            return false;
        }
        this->postprocess_pipeline.push_back(postprocess_node);
    }
    //申请输入和输出的指针空间
    int input_size = this->backend_model[0]->GetInputSize();
    int output_size = this->backend_model[0]->GetOutputSize();
    this->input_buffer = new float[input_size];
    this->output_buffer = new float[output_size];
    this->input_h = this->backend_model[0]->GetInputHeight();
    this->input_w = this->backend_model[0]->GetInputWidth();
    return true;

}
bool Pipeline::RunPipeline(std::string& image_path)
{
    cv::Mat image = cv::imread(image_path); 
    //循环调用已经注册的preprocess
    for (auto register_preprocess: preprocess_pipeline) {
		if(!register_preprocess->Run(image))
        {
            std::cerr << "preprocess_node: "<< register_preprocess->GetName() << "run failed" << '\n';
            return false;
        }
	}
    //将图片转成指针
 
    for (int i = 0; i < this->input_h * this->input_w; i++) {
        this->input_buffer[i] = image.at<cv::Vec3f>(i)[0];
        this->input_buffer[i + this->input_h * this->input_w] = image.at<cv::Vec3f>(i)[1];
        this->input_buffer[i + 2 * this->input_h* this->input_w] = image.at<cv::Vec3f>(i)[2];
    }

    //运行backend的推断
    if(!this->backend_model[0]->Infer(this->input_buffer,this->output_buffer))
    {
        std::cerr << "infer_node: "<< this->backend_model[0]->GetName() << "run failed" << '\n';
        return false;
    }

    //运行后处理
    if(!this->postprocess_pipeline[0]->Run(this->output_buffer))
    {
        std::cerr << "postprocess_node: "<< this->postprocess_pipeline[0]->GetName() << "run failed" << '\n';
        return false;
    }
    return true;
    
}
bool Pipeline::DestroyPipeline()
{
	for (auto& register_preprocess: preprocess_pipeline) {
		if (register_preprocess != nullptr)
		{
			delete register_preprocess;
		}
	}
	preprocess_pipeline.clear();
    for (auto& register_postprocess: postprocess_pipeline) {
		if (register_postprocess != nullptr)
		{
			delete register_postprocess;
		}
	}
	postprocess_pipeline.clear();
    for (auto& register_backend: backend_model) {
		if (register_backend != nullptr)
		{
            register_backend->Uninit();
			delete register_backend;
		}
	}
    backend_model.clear();
    if(input_buffer != nullptr)
    {
        delete input_buffer;
    }
    if(output_buffer != nullptr)
    {
        delete output_buffer;
    }
	return true;
}

bool Pipeline::PhaseJson(std::string& json_path, 
        std::vector<std::pair<std::string, MapCalcParam>>& preprocess_info,
        std::vector<std::pair<std::string, MapCalcParam>>& backend_info,
        std::vector<std::pair<std::string, MapCalcParam>>& postprocess_info)
{
    try
    {
        std::ifstream ifs;
        ifs.open(json_path);
        Json::Reader reader;
        Json::Value root;
        if (reader.parse(ifs, root))
        {
            Json::Value preprocess_info_json = root["Preprocess"];
            Json::Value backend_info_json = root["Infer"];
            Json::Value postprocess_info_json = root["Postprocess"];
            if(!this->PhaseParams(preprocess_info_json, preprocess_info))
            {
                std::cerr << "please check your preprocess" << '\n';
                return false;
            }
            if(!this->PhaseParams(backend_info_json, backend_info))
            {
                std::cerr << "please check your backend info" << '\n';
                return false;
            }
            if(!this->PhaseParams(postprocess_info_json, postprocess_info))
            {
                std::cerr << "please check your postprocess" << '\n';
                return false;
            }   

            return true;

        }
    }
    catch(...)
    {
        std::cerr << "this json can not open" << '\n';
    }
	return false;
}
bool Pipeline::PhaseParams(Json::Value& infer_info, std::vector<std::pair<std::string, MapCalcParam>>& info_list)
{
    try
    {
        int size = infer_info.size();
        for(int i=0 ; i < size; i++)
        {
            Json::Value info = infer_info[i];
            Json::Value type = info["type"];
            std::string type_value = type.asCString();
            Json::Value param_map = info["params"];
		    Json::Value::Members members;
            members = param_map.getMemberNames();
            MapCalcParam input;
            for (Json::Value::Members::iterator it = members.begin(); it != members.end(); it++)
            {
                std::string map_key = *it;
                Json::Value map_value = param_map[map_key];
                if (map_value.isBool())
                {
                    int value = map_value.asBool();
                    input.insert({ map_key,value });
                }
                else if (map_value.isInt())
                {
                    int value = map_value.asInt();
                    input.insert({ map_key,value });
                }
                else if (map_value.isDouble())
                {
                    double value = map_value.asDouble();
                    input.insert({ map_key,value });
                }
                else if (map_value.isString())
                {
                    std::string value = map_value.asCString();
                    input.insert({ map_key,value });

                }
                else if (map_value.isArray())
                {
                    std::vector<double> array;
                    int value_size = map_value.size();
                    for(int j=0; j < value_size; j++)
                    {
                        double value = map_value[j].asDouble();
                        array.push_back(value);
                    }
                    input.insert({ map_key,array});
                }
                else {
                    std::cerr << "this json can not open" << '\n';
                    return false;
                }
                
            }
            std::pair<std::string, MapCalcParam> single_info(type_value, input);
            info_list.push_back(single_info);
        }
    }
    catch(...)
    {
        return false;
    }
    return true;
    
}


} // namespace Engine