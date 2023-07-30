#ifndef ENGINE_backend_interface_H
#define ENGINE_backend_interface_H
#include "../utils/params_define.h"
#include "../utils/register.h"
namespace Engine
{

class BackendInterface{
public:
    /**
     * @brief backend的初始化的接口函数
     * 
     * @param backend_input 输出参数
     * @return true 
     * @return false 
     */
    virtual bool Init(MapCalcParam& backend_input) = 0;
    /**
     * @brief 不同 backend的推断主函数，由模版模式开发
     * 
     * @param inputs 输入指针
     * @param outputs 输出指针
     * @return true 
     * @return false 
     */
    virtual bool Infer(float* inputs, float* outputs) = 0;

    /**
     * @brief backend的销毁函数，由派生类实现
     * 
     * @return true 
     * @return false 
     */
    virtual bool Uninit() = 0;

    virtual int GetInputSize() = 0;
   
    virtual int GetOutputSize() = 0;

    virtual int GetInputHeight() = 0;
   
    virtual int GetInputWidth() = 0;
   
    virtual std::string GetName() = 0;

};
} // namespace Engine


#endif