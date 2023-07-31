#include "normalize_wrapper.h" 
namespace Engine
{
REGISTER_PREPROCESS(NormalizeWrapper)
bool NormalizeWrapper::Init(MapCalcParam& preprocess_input)
{
    try
    {
        
        _limit = GetParam(preprocess_input["limit"], _limit);
        _normalize = GetParam(preprocess_input["normalize"], _normalize);
        std::vector<double> mean = {0,0,0};
        mean = GetParam(preprocess_input["mean"], mean);
        _mean[0] = mean[0];
        _mean[1] = mean[1];
        _mean[2] = mean[2];
        
        std::vector<double> std = {0,0,0};
        std = GetParam(preprocess_input["std"], std);
        _std[0] = std[0];
        _std[1] = std[1];
        _std[2] = std[2];
        //检查size
        if ( _normalize && (_mean ==cv::Scalar(0,0,0) || _std == cv::Scalar(0,0,0)) )
        {
            std::cerr << 'please check the input of mean or std'  << '\n';
            return false;
        }
        // 输入参数
        
        std::cerr << "limit: " << _limit << '\n';
        std::cerr << "normalize: " << _normalize<< '\n';
        std::cerr << "mean: " << _mean << '\n';
        std::cerr << "std: " << _std << '\n';
        
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return true;
    
}
bool NormalizeWrapper::Run(cv::Mat& input_mat)
{
    input_mat.convertTo(input_mat, CV_32F);
    if(_limit)
    {
        input_mat = input_mat / 255.0;
       
    }
    if(_normalize)
    {
        cv::subtract(input_mat, _mean, input_mat);
	    cv::divide(input_mat, _std, input_mat);
    }
    return true;
    
}
    
};