#include "yolov5OrtPostprocess_wrapper.h" 
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
struct Detection
{
    cv::Rect box;
    float conf{};
    int classId{};
};

REGISTER_POSTPROCESS(Yolov5OrtPostprocessWrapper)
bool Yolov5OrtPostprocessWrapper::Init(MapCalcParam& postprocess_input)
{
    
  
    return true;
   
}
bool Yolov5OrtPostprocessWrapper::Run(float* output)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    int numClasses = 80;
    int elementsInBatch = 25200*85;
    double confThreshold = 0.3;
    double iouThreshold = 0.4;
    int scale_index = 0;
    int batch_size = 1;
    // only for batch size = 1
    for (auto it = 0; it != elementsInBatch; it += 85)
    {
        float clsConf = output[it + 4];

        if (clsConf > confThreshold)
        {
            int centerX = (int) (output[it +0]);
            int centerY = (int) (output[it +1]);
            int width = (int) (output[it + 2]);
            int height = (int) (output[it + 3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            float objConf;
            int classId;
            this->getBestClassInfo(&output[it], numClasses, objConf, classId);

            float confidence = clsConf * objConf;

            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);
    // std::cout << "amount of NMS indices: " << indices.size() << std::endl;

    std::vector<Detection> detections;

    for (int idx : indices)
    {
        Detection det;
        det.box = cv::Rect(boxes[idx]);
        // this->scaleCoords(resizedImageShape, det.box, originalImageShape);

        det.conf = confs[idx];
        det.classId = classIds[idx];
        detections.emplace_back(det);
    }
    return true;
    //return detections;
}

void Yolov5OrtPostprocessWrapper::getBestClassInfo(float* it, const int& numClasses,
                                    float& bestConf, int& bestClassId)
{
    // first 5 element are box and obj confidence
    bestClassId = 5;
    bestConf = 0;

    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }

}

void Yolov5OrtPostprocessWrapper::scaleCoords(const cv::Size& imageShape, cv::Rect& coords, const cv::Size& imageOriginalShape)
{
    float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
                          (float)imageShape.width / (float)imageOriginalShape.width);

    int pad[2] = {(int) (( (float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
                  (int) (( (float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f)};

    coords.x = (int) std::round(((float)(coords.x - pad[0]) / gain));
    coords.y = (int) std::round(((float)(coords.y - pad[1]) / gain));

    coords.width = (int) std::round(((float)coords.width / gain));
    coords.height = (int) std::round(((float)coords.height / gain));


}
std::vector<InstanceInfo> Yolov5OrtPostprocessWrapper::GetResult()
{
    return output_infos[0];
} 
};