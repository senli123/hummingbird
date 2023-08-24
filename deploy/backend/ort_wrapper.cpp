#include "ort_wrapper.h"

namespace Engine
{
REGISTER_BACKEND(OrtWrapper)
Ort::Env OrtEnv::env = nullptr;
bool OrtEnv::CreateEnv()
{
    if(env != nullptr)
    {
        return true;
    }
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    if (env == nullptr) {
        return false;
    }
    return true; 
}
bool OrtEnv::CreateSession(std::string& model_path, Ort::SessionOptions &session_options, Ort::Session &session)
{
    try
    {
        session = Ort::Session(this->env, model_path.c_str(), session_options);
    }
    catch(...)
    {
       return false;
    }
    if(session == nullptr)
    {
        return false; 
    }
    return true;
}
 
bool OrtEnv::DestroyEnv()
{
    return true;
}

bool OrtWrapper::Init(MapCalcParam& backend_input)
{
   
    // 解析输入的参数std::string engine_path;
    m_output_size = GetParam(backend_input["output_size"], m_output_size);
    m_model_path = GetParam(backend_input["model_path"], m_model_path);
    m_cuda_id = GetParam(backend_input["cuda_id"], m_cuda_id);
    // 输入参数
    std::cerr << "m_model_path: " << m_model_path <<'\n';
    std::cerr << "m_output_size: " << m_output_size <<'\n';
    std::cerr << "m_cuda_id: " << m_cuda_id <<'\n';

    if(!OrtEnv::get_instance().CreateEnv())
    {   
        std::cerr << "create env failed " << std::endl;
        return false;
    }

    this->sessionOptions = Ort::SessionOptions();    

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;
    bool isGPU = false;
    if(m_cuda_id != "cpu")
    {
        isGPU = true;
    }
    if (isGPU && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGPU && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: GPU" << std::endl;
        this->sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }

    if(!OrtEnv::get_instance().CreateSession(m_model_path, this->sessionOptions,this->session))
    {   
        std::cerr << "create session failed " << std::endl;
        return false;
    }

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    this->inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    
    
    // checking if width and height are dynamic
    if (this->inputTensorShape[2] == -1 && this->inputTensorShape[3] == -1)
    {
        std::cout << "Dynamic input shape" << std::endl;
        this->isDynamicInputShape = true;
    }

    for (auto shape : this->inputTensorShape)
        std::cout << "Input shape: " << shape << std::endl;

    this->m_batch_size = this->inputTensorShape[0];
    this->m_channels = this->inputTensorShape[1];
    this->m_input_h = this->inputTensorShape[2];
    this->m_input_w = this->inputTensorShape[3];

    for (const auto& element : inputTensorShape)
        this->inputTensorSize *= element;


    auto input_name = session.GetInputNameAllocated(0, allocator);
    std::cout << "Input name: " << input_name.get() << std::endl;
    //inputNames.push_back(input_name.get());
    char * strc = new char[strlen(input_name.get())+1];
    strcpy(strc, input_name.get());  
    this->inputNames.push_back(strc);

    auto output_name = session.GetOutputNameAllocated(0, allocator);
    std::cout << "Output name: " << output_name.get() << std::endl;
    //outputNames.push_back(output_name.get());
    char * strc1 = new char[strlen(output_name.get())+1];
    strcpy(strc1, output_name.get());  
    this->outputNames.push_back(strc1);

   this->memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    return true;
}


bool OrtWrapper::Uninit()
{
    return true;
} 


bool OrtWrapper::Infer(float* inputs, float* outputs)
{

    std::vector<float> inputTensorValues(inputs, inputs + this->inputTensorSize);

    std::vector<Ort::Value> inputTensors;

    

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            this->memoryInfo, inputTensorValues.data(), this->inputTensorSize,
            this->inputTensorShape.data(), this->inputTensorShape.size()
    ));

    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                              this->inputNames.data(),
                                                              inputTensors.data(),
                                                              1,
                                                              this->outputNames.data(),
                                                              1);
    auto* rawOutput =outputTensors[0].GetTensorData<float>();
    memcpy(outputs, rawOutput, m_output_size * sizeof(float));
    return true;
}

} // namespace Engine
