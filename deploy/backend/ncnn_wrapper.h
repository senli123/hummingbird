#pragma once
#ifndef ENGINE_ncnn_wrapper_H
#define ENGINE_ncnn_wrapper_H
#include <ncnn/layer.h>
#include <ncnn/net.h>
#include "backend_interface.h" 
namespace Engine
{

class NcnnWrapper : public BackendInterface{
public:
    virtual bool Init(MapCalcParam& backend_input);
    virtual bool Infer(float* inputs, float* outputs);
    virtual bool Uninit();
    virtual int GetInputSize()
    {
        return m_batch_size * m_channels * m_input_w * m_input_h;
    }
    virtual int GetOutputSize()
    {
        return m_output_size;
    }
    virtual int GetInputHeight()
    {
        return m_input_h; 
    }
   
    virtual int GetInputWidth()
    {
       return m_input_w;
    }
    virtual std::string GetName()
    {
        return "NcnnWrapper";
    }
private:
    int m_output_size;
    int m_input_h;
    int m_input_w;
    int m_batch_size = 1;
    int m_channels = 3;
    std::string m_input_name;  
    std::string m_output_name;
    std::string m_param_path;
    std::string m_bin_path;
    bool initialized_;
	ncnn::Net* net;

};

} // namespace Engine


#endif