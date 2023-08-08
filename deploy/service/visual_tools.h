#ifndef Engine_visualtools_h
#define Engine_visualtools_h
#include "../utils/params_define.h"
namespace Engine
{
    class VisualTools
{
public:
	~VisualTools() {}
	VisualTools(const VisualTools&) = delete;
	VisualTools& operator=(const VisualTools&) = delete;
	static VisualTools& get_instance() {
		static VisualTools instance;
		return instance;
	}
    //先完成单batch
    void detectionVisual(cv::Mat& image, int resize_w, int resize_h,std::vector<InstanceInfo> &output_infos);
   
    
private:
	VisualTools() {};
    bool Update_coords(int img_width, int img_height, int resize_w, int resize_h, cv::Rect &rect);

};
} // namespace Engine



#endif