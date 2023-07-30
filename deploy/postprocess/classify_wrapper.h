#ifndef ENGINE_reszie_wrapper_H
#define ENGINE_resize_wrapper_H
#include "postprocess_interface.h" 
namespace Engine
{
class ClassifyWrapper : public PostprocessInterface{
public:
    virtual bool Init(MapCalcParam& postprocess_input);
    virtual bool Run(float* output);
    virtual std::string GetName()
    {
        return "ClassifyWrapper";
    }

private:
    int _class_nums = 0;  
    int _top_nums = 0;
   
};

} // namespace Engine


#endif