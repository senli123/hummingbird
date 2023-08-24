#ifndef ENGINE_tensorrt_wrapper_H
#define ENGINE_tensorrt_wrapper_H
#include <onnxruntime_cxx_api.h>
#include <utility>
#include <chrono>
#include "backend_interface.h" 
namespace Engine
{
class OrtEnv
{
public:
	~OrtEnv() {}
	OrtEnv(const OrtEnv&) = delete;
	OrtEnv& operator=(const OrtEnv&) = delete;
	static OrtEnv& get_instance() {
		static OrtEnv instance;
		return instance;
	}
    
    bool CreateEnv();
    bool CreateSession(std::string& model_path, Ort::SessionOptions &session_options,  Ort::Session &session);
    bool DestroyEnv();
private:
	OrtEnv() {};

private:
    
    static Ort::Env env;
};

class OrtWrapper : public BackendInterface{
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
        return "OrtWrapper";
    }
private:
    int m_input_h;
    int m_input_w;
    int m_batch_size = 1;
    int m_channels = 3;
    Ort::MemoryInfo memoryInfo{nullptr};
    std::vector<int64_t> inputTensorShape;
    size_t inputTensorSize = 1;
    std::string m_model_path;
    std::string m_cuda_id;
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};

    int m_output_size;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    bool isDynamicInputShape = false;
};

} // namespace Engine


#endif