#include "tensorrt_wrapper.h"

namespace Engine
{
REGISTER_BACKEND(TensorrtWrapper)
nvinfer1::IRuntime* TensorrtEnv::m_runtime = nullptr;
bool TensorrtEnv::CreateEnv()
{
    if(m_runtime != nullptr)
    {
        return true;
    }
    static Logger gLogger;
    m_runtime = nvinfer1::createInferRuntime(gLogger);
    if (m_runtime == nullptr) {
        return false;
    }
    return true; 
}
nvinfer1::ICudaEngine* TensorrtEnv::CreateEngine(void const* trtModelStream, std::size_t size)
{
    return m_runtime->deserializeCudaEngine(trtModelStream, size);
}
 
bool TensorrtEnv::DestroyEnv()
{
    if(m_runtime != nullptr)
    {
        m_runtime->destroy();
    }
    return true;
}

bool TensorrtWrapper::Init(MapCalcParam& backend_input)
{
   
    // 解析输入的参数std::string engine_path;
    m_input_h = GetParam(backend_input["input_h"], m_input_h);  
    m_input_w = GetParam(backend_input["input_w"], m_input_w); 
    m_output_size = GetParam(backend_input["output_size"], m_output_size);
    m_input_name = GetParam(backend_input["input_name"], m_input_name);  
    m_output_name = GetParam(backend_input["output_name"], m_output_name); 
    m_engine_path = GetParam(backend_input["engine_path"], m_engine_path);
    m_cuda_id = GetParam(backend_input["cuda_id"], m_cuda_id);
    cudaSetDevice(m_cuda_id);
    // 输入参数
    std::cerr << "m_input_h: " << m_input_h <<'\n';
    std::cerr << "m_input_w: " << m_input_w <<'\n';
    std::cerr << "m_output_size: " << m_output_size <<'\n';
    std::cerr << "m_input_name: " << m_input_name <<'\n';
    std::cerr << "m_output_name: " << m_output_name <<'\n';
    std::cerr << "m_engine_path: " << m_engine_path <<'\n';
    std::cerr << "m_cuda_id: " << m_cuda_id <<'\n';
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file(m_engine_path, std::ios::binary);
    if (!file.good())
    {
        std::cerr << "file is not good" << std::endl;
        return false;
    }
  
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    if(trtModelStream == nullptr)
    {
        std::cerr << "trtModelStream == nullptr" << std::endl;
        return false;
    }
    
    file.read(trtModelStream, size);
    file.close();

    if(!TensorrtEnv::get_instance().CreateEnv())
    {   
        std::cerr << "create runtime failed " << std::endl;
        return false;
    }
    m_engine = TensorrtEnv::get_instance().CreateEngine(trtModelStream, size);
    if (m_engine == nullptr) {
        std::cerr << "m_engine == nullptr" << std::endl;
        return false;
    }

    m_context = m_engine->createExecutionContext();
    if (m_context == nullptr) {
        std::cerr << "m_context == nullptr" << std::endl;
        return false;
    }

    delete[] trtModelStream;

    // Allocate input and output buffer;
    const int m_input_size = m_input_h * m_input_w * m_batch_size * m_channels;
    m_input_cpu_buffer = new float[m_input_size];
    m_output_cpu_buffer = new float[m_output_size];
    if (m_input_cpu_buffer == nullptr || m_output_cpu_buffer == nullptr) {
        std::cerr << "m_input_cpu_buffer == nullptr || m_output_cpu_buffer == nullptr" << std::endl;
        return false;
    }
    if(m_engine->getNbBindings() != 2) {
        std::cerr << "m_engine->getNbBindings() != 2" << std::endl;
        return false;
    }

    m_input_index = m_engine->getBindingIndex(m_input_name.c_str());
    m_output_index = m_engine->getBindingIndex(m_output_name.c_str());

    CUDA_CHECK(cudaMalloc(&gpu_buffers[m_input_index], m_input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_buffers[m_output_index], m_output_size * sizeof(float)));

    CUDA_CHECK(cudaStreamCreate(&m_stream));
    delete[] m_input_cpu_buffer;
    delete[] m_output_cpu_buffer;

    return true;
}


bool TensorrtWrapper::Uninit()
{
    
    cudaStreamDestroy(m_stream);
    CUDA_CHECK(cudaFree(gpu_buffers[m_input_index]));
    CUDA_CHECK(cudaFree(gpu_buffers[m_output_index]));
    m_context->destroy();
    m_engine->destroy();
    TensorrtEnv::get_instance().DestroyEnv();
    return true;
} 


bool TensorrtWrapper::Infer(float* inputs, float* outputs)
{
    cudaSetDevice(m_cuda_id);
    CUDA_CHECK(cudaMemcpyAsync(gpu_buffers[m_input_index], inputs, m_input_h * m_input_w * m_channels * m_batch_size * sizeof(float), cudaMemcpyHostToDevice, m_stream));
    m_context->enqueue(m_batch_size, gpu_buffers, m_stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(outputs, gpu_buffers[m_output_index], m_output_size * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
    cudaStreamSynchronize(m_stream);
    return true;
}

} // namespace Engine
