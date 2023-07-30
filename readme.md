# 转换成不同ir中间格式的json定义
* onnx
```json
{
"Convert":{
        "type":"torch2onnx",
        "params":{
             "pth_path": "",
            "input_shape": "",
               "ir_path":"",
               "input_names": "",
               "output_names": "",
               "dynamic_axes":,
               "verbose":"",
        }  
    }
}
```

* torchscript
```json
{
"Convert":{
        "type":"torch2torchscript",
        "params":{
             "pth_path": "",
            "input_shape": "",
               "ir_path":""
        }  
    }
}
```

* wts
```json
{
"Convert":{
        "type":"torch2wts",
        "params":{
             "pth_path": "",
            "input_shape": "",
               "ir_path":"",
               "cuda_id_str":"cuda:0"
        }  
    }
}

```

# 不同backend转换模型ir的json定义
* onnx2tensorrt
```json
{
"BackendConvert":{
        "type":"onnx2tensorrt",
        "params":{
            "onnx_path":"",
            "output_file_prefix":"",
            "input_shapes":,
            "max_workspace_size":, 
                    "device_id":, 
                    "fp16_mode":, 
                    "int8_mode":, 
                    "int8_param":, 
        } 
    }
}
                   
                    
```

* wts2tensorrt
```json
{
"BackendConvert":{
        "type":"wts2tensorrt",
        "params":{
            "name":"",
            "wts_path":"",
            "output_file_prefix":"",
            "input_blob_name":,
            "output_blob_name":, 
            "max_batch_size":, 
            "input_h":, 
            "input_w":
                    
        } 
    }
}
```