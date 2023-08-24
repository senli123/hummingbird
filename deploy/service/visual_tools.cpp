#include "visual_tools.h"
namespace Engine
{
void VisualTools::detectionVisual(cv::Mat& image, int resize_w, int resize_h, std::vector<InstanceInfo> &output_infos)
{
    int img_width = image.cols;
    int img_height = image.rows;
    //循环所有检测结果进行左边转换然后绘制输出
    for ( auto single_result : output_infos)
    {
        cv::Rect rect = single_result.rect; //(x,y,w,h)左上和长宽
	    float score = single_result.score;            
	    int class_id = single_result.class_id; 
        this->Update_coords(img_width, img_height, resize_w, resize_h, rect);
        //输出
        std::cerr << "------------------------"<< '\n';
        std::cerr << "rect(x,y,w,h): "<< rect.x  << ","<< rect.y << ","<< rect.width << ","<< rect.height << '\n';
        std::cerr << "score: "<< score << '\n';
        std::cerr << "class_id: "<< class_id << '\n';
        
        cv::rectangle(image, rect, cv::Scalar(255, 0, 0), 2, cv::LINE_8, 0);
        //类别坐标和socre坐标
		cv::Point class_point, score_point;
        class_point.x = rect.x - 10;
		class_point.y = rect.y - 20;
		score_point.x = rect.x - 10;
		score_point.y = rect.y - 10;
        cv::putText(image, std::to_string(class_id), class_point, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, 8, 0);
		cv::putText(image, std::to_string(score), score_point, cv::FONT_HERSHEY_COMPLEX, 0.5,cv::Scalar(0, 0, 255), 1, 8, 0);
    }
    cv::imwrite("/workspace/lisen/tensorrt/my_tensorrt/img/output/bus1.jpg",image);
}

bool VisualTools::Update_coords(int img_width, int img_height, int resize_w, int resize_h, cv::Rect &rect)
{
    try
    {
        float x1 = rect.x;
        float y1 = rect.y;
        float x2 = rect.x + rect.width;
        float y2 = rect.y + rect.height;
        float gain_x = float(resize_w) /float(img_width);
        float gain_y = float(resize_h) /float(img_height);
        float gain = std::min(gain_x,gain_y);
        float pad_w = (resize_w - img_width * gain) / 2.0;
	    float pad_h = (resize_h - img_height * gain) / 2.0;
        x1 -= pad_w;
        y1 -= pad_h;
        x2 -= pad_w;
        y2 -= pad_h;
        x1 /= gain;
        y1 /= gain;
        x2 /= gain;
        y2 /= gain;
        x1 = std::max(float(0),x1);
        y1 = std::max(float(0),y1);
        x2 = std::min(float(img_width -1),x2);
        y2 = std::min(float(img_height-1),y2);
        rect.x = x1;
        rect.y = y1;
        rect.width = x2-x1;
        rect.height = y2-y1;
    }
    catch(const std::exception& e)
    {
        return false;
    }
    return true;

}
}