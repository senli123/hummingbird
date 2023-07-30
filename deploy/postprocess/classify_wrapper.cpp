#include "classify_wrapper.h" 
namespace Engine
{
REGISTER_POSTPROCESS(ClassifyWrapper)
bool ClassifyWrapper::Init(MapCalcParam& postprocess_input)
{
    try
    {
        _class_nums = GetParam(postprocess_input["class_nums"], _class_nums);  
        _top_nums = GetParam(postprocess_input["top_nums"], _top_nums);
        if (_class_nums == 0 || _top_nums == 0)
        {
            std::cerr << 'please check the input of class_nums or top_nums'  << '\n';
            return false;
        }
        // 输入参数
        std::cerr << "class_nums: " << _class_nums <<'\n';
        std::cerr << "top_nums: " << _top_nums <<'\n';
        
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return true;
    
}
bool ClassifyWrapper::Run(float* output)
{
    std::vector<std::pair<float, int>> index_info;
    for (int class_index = 0; class_index < this->_class_nums; class_index++)
    {
        index_info.push_back(std::make_pair(output[class_index],class_index));
    }
    std::partial_sort(index_info.begin(), index_info.begin() + _top_nums, index_info.end(),
            std::greater<std::pair<float, int>>());
    for (int i = 0; i < this->_top_nums; i++)
    {
        std::cerr << "top index: "<< i+1 <<'\n';
        std::cerr << "class score: "<< index_info[i].first <<'\n';
        std::cerr << "class index: "<< index_info[i].second <<'\n';
        std::cerr << "-----------------------------"<<'\n';
    }
    return true;
}
    
};