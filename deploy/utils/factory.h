#ifndef ENGINE_factory_H
#define ENGINE_factory_H
#include <iostream> 
#include <map>
#include <string>
namespace Engine
{
typedef void* (*register_func)(); //类指针

class Store
{
public:
    /**
     * @brief 返回静态工厂实例
     * 
     * @return Store* 
     */
    static Store* getInstance();

    /**
     * @brief 查找backend类
     * 
     * @param backend_name 已注册的backend的name
     * @return void* 返回对应的backend类
     */
    void* findBackendInstance(const std::string& backend_name);
    /**
     * @brief 注册指定的backend类
     * 
     * @param backend_name backend的对应名字
     * @param backend_func 要注册backend的类
     */
    void registerBackendInstance(const std::string& backend_name, register_func backend_func);
    /**
     * @brief 输出工厂中所有注册的backend类的name
     * 
     */
    void getBackendKeys();


    //Preprocess
    /**
     * @brief 查找工厂中已经注册的preprocess类
     * 
     * @param preprocess_name preprocesss的name
     * @return void* 
     */
    void* findPreprocessInstance(const std::string& preprocess_name);
    /**
     * @brief 注册preprocess类
     * 
     * @param preprocess_name 要注册的preprocess的name
     * @param preprocess_func 要注册preprocess类
     */
    
    void registerPreprocessInstance(const std::string& preprocess_name, register_func preprocess_func);
    
    /**
     * @brief 输出工厂中所有注册的preprocess类的name
     * 
     */
    void getPreprocessKeys();
    
    //Postprocess
    /**
     * @brief 查找工厂中已经注册的postprocess类
     * 
     * @param postprocess_name postprocesss的name
     * @return void* 
     */
    void* findPostprocessInstance(const std::string& postprocess_name);

     /**
     * @brief 注册postprocess类
     * 
     * @param postprocess_name 要注册的postprocess的name
     * @param postprocess_func 要注册postprocess类
     */
    void registerPostprocessInstance(const std::string& postprocess_name, register_func postprocess_func);
    
    /**
     * @brief 输出工厂中所有注册的postprocess类的name
     * 
     */
    void getPostprocessKeys();
    
private:
    /// @brief 保存所有backend类的字典
    std::map<std::string, register_func> m_register;
    /// @brief 保存所有preprocess类的字典
    std::map<std::string, register_func> m_register_preprocess;
    /// @brief 保存所有postprocess类的字典
    std::map<std::string, register_func> m_register_postprocess;
    /// @brief 工厂实例
    static Store* instance;  
};   
} // namespace Engine


#endif