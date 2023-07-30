#include "resize_wrapper.h" 
namespace Engine
{
REGISTER_PREPROCESS(ResizeWrapper)
bool ResizeWrapper::Init(MapCalcParam& preprocess_input)
{
    try
    {
        _bgr2rgb = GetParam(preprocess_input["bgr2rgb"], _bgr2rgb);
        _keep_ratio = GetParam(preprocess_input["keep_ratio"], _keep_ratio);
        std::vector<double> size = {0,0};
        size = GetParam(preprocess_input["size"], size);
        _size.width = size[0];
        _size.height = size[1];
        
        //检查size
        if (_size == cv::Size(0,0))
        {
            std::cerr << 'please check the input of size'  << '\n';
            return false;
        }
        // 输入参数
        std::cerr << "bgr2rgb: " << _bgr2rgb;
        std::cerr << "keep_ratio: " << _keep_ratio;
        std::cerr << "size: " << _size;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return true;
    
}
bool ResizeWrapper::Run(cv::Mat& input_mat)
{
    if(_bgr2rgb)
    {
        cv::cvtColor(input_mat, input_mat, cv::COLOR_BGR2RGB);
    }
    cv::resize(input_mat, input_mat, _size, 0, 0, cv::INTER_LINEAR);
    return true;
}
    
};