#include "factory.h"
namespace Engine
{
Store* Store::instance = nullptr;

Store* Store::getInstance()
{
    if (!instance) instance = new Store;
    return instance;
}

void* Store::findBackendInstance(const std::string& backend_name)
{
    std::map < std::string, register_func>::iterator it = m_register.find(backend_name);
    if (it == m_register.end())
    {
        return nullptr;
    }
    else
    {
        return it->second();
    }
}
void Store::registerBackendInstance(const std::string& backend_name, register_func backend_func)
{
    m_register[backend_name] = backend_func;
}
void Store::getBackendKeys()
{
    for (std::map<std::string, register_func>::iterator it = m_register.begin(); it != m_register.end(); ++it)
    {
        std::cout<<it->first<<", ";
    }
    std::cout<<std::endl;
}


void* Store::findPreprocessInstance(const std::string& preprocess_name)
{
    std::map < std::string, register_func>::iterator it = m_register_preprocess.find(preprocess_name);
    if (it == m_register.end())
    {
        return nullptr;
    }
    else
    {
        return it->second();
    }
}

void Store::registerPreprocessInstance(const std::string& preprocess_name, register_func preprocess_func)
{
    m_register_preprocess[preprocess_name] = preprocess_func;
}

void Store::getPreprocessKeys()
{
    for (std::map<std::string, register_func>::iterator it = m_register_preprocess.begin(); it != m_register.end(); ++it)
    {
        std::cout<<it->first<<", ";
    }
    std::cout<<std::endl;
}
    //Postprocess
void* Store::findPostprocessInstance(const std::string& postprocess_name)
{
    std::map < std::string, register_func>::iterator it = m_register_postprocess.find(postprocess_name);
    if (it == m_register.end())
    {
        return nullptr;
    }
    else
    {
        return it->second();
    }
}

void Store::registerPostprocessInstance(const std::string& postprocess_name, register_func postprocess_func)
{
    m_register_preprocess[postprocess_name] = postprocess_func;
}

void Store::getPostprocessKeys()
{
    for (std::map<std::string, register_func>::iterator it = m_register_postprocess.begin(); it != m_register.end(); ++it)
    {
        std::cout<<it->first<<", ";
    }
    std::cout<<std::endl;
}

} // namespace Engine

