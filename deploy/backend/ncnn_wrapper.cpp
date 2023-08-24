#include "ncnn_wrapper.h"

namespace Engine
{
REGISTER_BACKEND(NcnnWrapper)
bool NcnnWrapper::Init(MapCalcParam& backend_input)
{
   try
   {
        // 解析输入的参数std::string engine_path;
        m_param_path = GetParam(backend_input["param_path"], m_param_path);
        m_bin_path = GetParam(backend_input["bin_path"], m_bin_path);
        m_output_size = GetParam(backend_input["output_size"], m_output_size);
        m_input_h = GetParam(backend_input["input_h"], m_input_h);  
        m_input_w = GetParam(backend_input["input_w"], m_input_w); 
        m_input_name = GetParam(backend_input["input_name"], m_input_name);  
        m_output_name = GetParam(backend_input["output_name"], m_output_name); 
        // 输入参数
        std::cerr << "m_input_h: " << m_input_h <<'\n';
        std::cerr << "m_input_w: " << m_input_w <<'\n';
        std::cerr << "m_output_size: " << m_output_size <<'\n';
        std::cerr << "m_param_path: " << m_param_path <<'\n';
        std::cerr << "m_bin_path: " << m_bin_path <<'\n';
        this->net = new ncnn::Net();
        this->net->load_param(m_param_path.c_str());
        this->net->load_model(m_bin_path.c_str());
        initialized_ = true;
   }
   catch(...)
   {
        return false;
   }
    return true; 
}

bool NcnnWrapper::Uninit()
{
    
    if (this->net) {
		this->net->clear();
	}
    return true;
} 


bool NcnnWrapper::Infer(float* inputs, float* outputs)
{
    try
    {
        ncnn::Mat in;
        in.from_pixels((unsigned char*)inputs, ncnn::Mat::PIXEL_RGB, m_input_h, m_input_w);
        std::vector<ncnn::Mat> out;
        ncnn::Extractor ex = this->net->create_extractor();
        // ex.input(this->Nparams.input_name, in);
        // for (size_t i = 0; i < this->Nparams.output_name.size(); i++)
        // {
        //     ncnn::Mat temp_out;
        //     ex.extract(this->Nparams.output_name[i], temp_out);
        //     out.push_back(temp_out);
        // }
        // ncnn::Mat output = out[0];
        ncnn::Mat temp_out;
        ex.input(m_input_name.c_str(), in);
        ex.extract(m_output_name.c_str(), temp_out);
        memcpy(outputs, temp_out.data, m_output_size * sizeof(float));
    }
    catch(...)
    {
        return false;
    }
    return true;
}
}