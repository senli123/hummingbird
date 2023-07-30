#ifndef ENGINE_create_interface_H
#define ENGINE_create_interface_H
#include "../utils/factory.h"
#include "../preprocess/preprocess_interface.h"
#include "../backend/backend_interface.h"
#include "../postprocess/postprocess_interface.h"
namespace Engine
{
  
class PreprocessFactory
{
public:
  static PreprocessInterface* createPreprocessInterface(const std::string& name)
  {
    return (PreprocessInterface*)Store::getInstance()->findPreprocessInstance(name);
  }
};

class BackendFactory
{
public:
  static BackendInterface* createBackendInterface(const std::string& name)
  {
    return (BackendInterface*)Store::getInstance()->findBackendInstance(name);
  }
};

class PostprocessFactory
{
public:
  static PostprocessInterface* createPostprocessInterface(const std::string& name)
  {
    return (PostprocessInterface*)Store::getInstance()->findPostprocessInstance(name);
  }
};  
} // namespace Engine



#endif