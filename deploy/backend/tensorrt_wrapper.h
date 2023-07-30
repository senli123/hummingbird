#ifndef ENGINE_tensorrt_wrapper_H
#define ENGINE_tensorrt_wrapper_H
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "logging.h"
#include <fstream>
#include <map>
#include <chrono>
#include "backend_interface.h" 
namespace Engine
{
#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK
class TensorrtEnv
{
public:
	~TensorrtEnv() {}
	TensorrtEnv(const TensorrtEnv&) = delete;
	TensorrtEnv& operator=(const TensorrtEnv&) = delete;
	static TensorrtEnv& get_instance() {
		static TensorrtEnv instance;
		return instance;
	}
    bool CreateEnv();
    nvinfer1::ICudaEngine* CreateEngine(void const* trtModelStream, std::size_t size);
    bool DestroyEnv();
private:
	TensorrtEnv() {};

private:
    
    static nvinfer1::IRuntime* m_runtime;
};

class TensorrtWrapper : public BackendInterface{
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
        return "TensorrtWrapper";
    }
private:
    int m_input_h;
    int m_input_w;
    int m_batch_size = 1;
    int m_channels = 3;
    int m_output_size;
    std::string m_input_name;
    std::string m_output_name;
    std::string m_engine_path;
    int m_cuda_id = 0;
    // Tensorrt engine
    nvinfer1::ICudaEngine* m_engine;
    // Tensorrt execution context
    nvinfer1::IExecutionContext* m_context;

    // Tensorrt model stream
    cudaStream_t m_stream;

    // Tensorrt model input cpu buffer
    float* m_input_cpu_buffer;

    // Tensorrt model output cpu buffer
    float* m_output_cpu_buffer;

    // Tensorrt model input gpu buffer
    void* gpu_buffers[2];

    int m_input_index;
    int m_output_index;
    
};

} // namespace Engine


#endif