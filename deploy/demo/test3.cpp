#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <cuda_runtime_api.h>
#include <cvcuda/OpConvertTo.hpp>
#include <cvcuda/OpNormalize.hpp>
#include <cvcuda/OpReformat.hpp>
#include <cvcuda/OpResize.hpp>
#include <math.h>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
 int main(){
    const int batch_size = 1;
const int width = 640;
const int height = 640;
    // Resize to the dimensions of input layer of network
    nvcv::Tensor resizedTensor(batch_size, {width, height}, nvcv::FMT_BGR8);
    cvcuda::Resize resizeOp;
    resizeOp(stream, input_image_tensor, resizedTensor, NVCV_INTERP_LINEAR);

    // convert BGR to RGB
    nvcv::Tensor rgbTensor(batch_size, {width, height}, nvcv::FMT_RGB8);
    cvcuda::CvtColor cvtColorOp;
    cvtColorOp(stream, resizedTensor, rgbTensor, NVCV_COLOR_BGR2RGB);

    // Convert to data format expected by network (F32). Apply scale 1/255.
    nvcv::Tensor floatTensor(batch_size, {width, height}, nvcv::FMT_RGBf32);
    cvcuda::ConvertTo convertOp;
    convertOp(stream, rgbTensor, floatTensor, 1.0 / 255.0, 0.0);

    // Convert the data layout from HWC to CHW
    cvcuda::Reformat reformatOp;
    reformatOp(stream, floatTensor, model_input_tensor);
    return 0;
}