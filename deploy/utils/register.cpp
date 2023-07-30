#include "register.h"
namespace Engine
{
RegisterBackend::RegisterBackend(std::string backend_name, register_func backend_func)
{
  Store::getInstance()->registerBackendInstance(backend_name, backend_func);
}

RegisterBackend::~RegisterBackend() {}


RegisterPreprocess::RegisterPreprocess(std::string preprocess_name, register_func preprocess_func)
{
  Store::getInstance()->registerPreprocessInstance(preprocess_name, preprocess_func);
}
RegisterPreprocess::~RegisterPreprocess(){}

RegisterPostprocess::RegisterPostprocess(std::string postprocess_name, register_func postprocess_func)
{
  Store::getInstance()->registerPreprocessInstance(postprocess_name, postprocess_func);
}
RegisterPostprocess::~RegisterPostprocess(){}

} // namespace Engine
