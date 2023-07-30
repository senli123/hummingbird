#ifndef Engine_register_h
#define Engine_register_h
#include "factory.h"

namespace Engine
{
/**
 * @brief 注册backend的宏定义
 * 
 */
#define REGISTER_BACKEND(backend_name) \
    class backend_name##Register \
    { \
    public: \
        static void* newInstance() \
        { \
            return new backend_name; \
        } \
    private: \
        static const RegisterBackend reg; \
    }; \
const RegisterBackend backend_name##Register::reg(#backend_name, backend_name##Register::newInstance);

/**
 * @brief 注册preprocess的宏定义
 * 
 */
#define REGISTER_PREPROCESS(preprocess_name) \
    class preprocess_name##Register \
    { \
    public: \
        static void* newInstance() \
        { \
            return new preprocess_name; \
        } \
    private: \
        static const RegisterPreprocess reg; \
    }; \
const RegisterPreprocess preprocess_name##Register::reg(#preprocess_name, preprocess_name##Register::newInstance);

/**
 * @brief 注册postprocess的宏定义
 * 
 */
#define REGISTER_POSTPROCESS(postprocess_name) \
    class postprocess_name##Register \
    { \
    public: \
        static void* newInstance() \
        { \
            return new postprocess_name; \
        } \
    private: \
        static const RegisterPostprocess reg; \
    }; \
const RegisterPostprocess postprocess_name##Register::reg(#postprocess_name, postprocess_name##Register::newInstance);

class RegisterBackend
{
public:
    /**
     * @brief backend的注册函数
     * 
     * @param backend_name backend的name
     * @param backend_func bakcend对应的类
     */
    RegisterBackend(std::string backend_name, register_func backend_func);
    ~RegisterBackend();
};

class RegisterPreprocess
{
public:
    /**
     * @brief preprocess的注册函数
     * 
     * @param RegisterPreprocess preprocess的name
     * @param RegisterPreprocess preprocess对应的类
     */
    RegisterPreprocess(std::string preprocess_name, register_func preprocess_func);
    ~RegisterPreprocess();
};

class RegisterPostprocess
{
public:
    /**
     * @brief postprocess的注册函数
     * 
     * @param postprocess_name postprocess的name
     * @param postprocess_func postprocess对应的类
     */
    RegisterPostprocess(std::string postprocess_name, register_func postprocess_func);
    ~RegisterPostprocess();
};

} // namespace Engine



#endif