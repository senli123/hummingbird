#include "yolov5Postprocess_wrapper.h" 
#include <opencv2/opencv.hpp>
namespace Engine
{
static bool cmp(const ClassInfo& a, const ClassInfo& b) {
  return a.o_rect_cof > b.o_rect_cof;
}

static float iou(cv::Rect lbox , cv::Rect rbox) {
  float interBox[] = {
    (std::max)(lbox.x , rbox.x), //left
    (std::min)(lbox.x + lbox.width, rbox.x + rbox.width), //right
    (std::max)(lbox.y, rbox.y), //top
    (std::min)(lbox.y + lbox.height, rbox.y + rbox.height), //bottom
  };

  if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
    return 0.0f;

  float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
  return interBoxS / (lbox.width * lbox.height + rbox.width * rbox.height - interBoxS);
}
REGISTER_POSTPROCESS(Yolov5PostprocessWrapper)
bool Yolov5PostprocessWrapper::Init(MapCalcParam& postprocess_input)
{
    
  
    return true;
   
}
bool Yolov5PostprocessWrapper::Run(float* output)
{
    //输出的output的格式为[batch_size,25200,85]
    bool err;
    double confthre = 0.25;
    double iou_thresh = 0.45;
    int scale_index = 0;
    int batch_size = 1;
    //构造结构体，用来对后面进行nms操作
    std::vector<std::map<int,std::vector<ClassInfo>>> classinfo(batch_size);
    for (int batch_index = 0; batch_index < batch_size; batch_index++)
    {
        int cur_batch_start_index = batch_index * 25200*85;
        //循环所有的框
        for(int bbox_index = 0; bbox_index < 25200; bbox_index++)
        {
            int cur_bbox_index = cur_batch_start_index + bbox_index* 85;
            //先得到当前框的score，进行第一次筛选
            double box_prob = output[cur_bbox_index + 4];
            if (box_prob<confthre)
            {
                continue;
            }
            //二次筛选,得到最大值和对应的index
            double max_score = 0.0;
            int max_index = 0;
            for(int i = 0; i < 80; i++)
            {
                double cur_score = output[cur_bbox_index + 5 + i];
                if (cur_score > max_score)
                {
                    max_score = cur_score;
                    max_index = i;
                }
            }
            //如果框的score和class_score乘积的低分小于阈值则跳过
            float cof = box_prob * max_score;
            //对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
            if(cof<confthre)
            {
                continue;
            }
            //进行坐标转换
            double x_c =output[cur_bbox_index + 0];
            double y_c =output[cur_bbox_index + 1];
            double w =output[cur_bbox_index + 2];
            double h =output[cur_bbox_index + 3];
            double r_x = x_c - w/2;
            double r_y = y_c - h/2;
            cv::Rect rect = cv::Rect(round(r_x),round(r_y),round(w),round(h));
            if (classinfo[batch_index].find(max_index)!=classinfo[batch_index].end())
            {
                ClassInfo cur_info = {rect, cof};
                classinfo[batch_index][max_index].push_back(cur_info);
                
            }else{
                ClassInfo cur_info = {rect, cof};
                classinfo[batch_index].insert(std::pair<int,std::vector<ClassInfo>>(max_index,{cur_info}));

            }
        
        }
        
    }
    //保存最终的输出
    this->output_infos.clear(); //每次跑后处理都要清空一次结果
    for(auto one_batch_result: classinfo)
    {
        std::vector<InstanceInfo> one_batch_output;
        for(auto cur_class_result : one_batch_result)
        {
            int class_index = cur_class_result.first;
            auto& dets = cur_class_result.second;
            std::sort(dets.begin(), dets.end(), cmp);
            for (size_t m = 0; m < dets.size(); ++m) {
                auto& item = dets[m];
                InstanceInfo cur_InstanceInfo = {item.o_rect,item.o_rect_cof,class_index};
                one_batch_output.push_back(cur_InstanceInfo);
                for (size_t n = m + 1; n < dets.size(); ++n) {
                    if (iou(item.o_rect, dets[n].o_rect) > iou_thresh) {
                        dets.erase(dets.begin() + n);
                    --n;
                    }
                }
      
            }
        
        }
        this->output_infos.push_back(one_batch_output);
    }
    return true;
}
std::vector<InstanceInfo> Yolov5PostprocessWrapper::GetResult()
{
    return output_infos[0];
} 
};