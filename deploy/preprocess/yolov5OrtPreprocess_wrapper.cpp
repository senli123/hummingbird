// #include "yolov5OrtPreprocess_wrapper.h" 
// namespace Engine
// {
// REGISTER_PREPROCESS(Yolov5OrtPreprocessWrapper)
// bool Yolov5OrtPreprocessWrapper::Init(MapCalcParam& preprocess_input)
// {
     
//     return true;
    
// }
// bool Yolov5OrtPreprocessWrapper::Run(cv::Mat& input_mat)
// {
//     int input_h = 640;  
//     int input_w = 640;
//     std::vector<int> color = {114,114,114};
//     int stride = 32;
//     cv::Mat rgb_img;
//     cv::cvtColor(input_mat, rgb_img, cv::COLOR_BGR2RGB);
    
//     int ih = input_mat.rows;
//     int iw = input_mat.cols;
//     float scale = std::min(static_cast<float>(input_w) / static_cast<float>(iw), static_cast<float>(input_h) / static_cast<float>(ih));
    
//     int new_unpad_w = int(round(iw*scale));
//     int new_unpad_h = int(round(ih*scale)); 

//     int nh = static_cast<int>(scale * static_cast<float>(ih));
//     int nw = static_cast<int>(scale * static_cast<float>(iw));
//     int dh = (input_h - nh) / 2;
//     int dw = (input_w - nw) / 2;
//     cv::Mat rgb_resize_img;
//     cv::resize(rgb_img, rgb_resize_img, cv::Size(new_unpad_w, new_unpad_h));
//     int top =  int(round(dh - 0.1));
//     int bottom = int(round(dh + 0.1));
//     int left = int(round(dw - 0.1));
//     int right = int(round(dw + 0.1));
//     cv::copyMakeBorder(rgb_resize_img, rgb_resize_img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(128,128,128));
    
//     rgb_resize_img.convertTo(rgb_resize_img,CV_32F);
//     input_mat = rgb_resize_img/255.0f;
//     return true;
// }
    
// };