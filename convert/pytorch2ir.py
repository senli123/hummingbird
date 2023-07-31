"""torch转换成onnx、torchscript和wts的封装
"""
from typing import Any, Dict, Optional, Sequence, Tuple, Union,OrderedDict
import torch
import struct
from torchsummary import summary
class Pytorch2ir():
    def torch2onnx(self, pth_path: str,
               input_shape: Sequence[int],
               ir_path:str,
               input_names: Optional[Sequence[str]] = None,
               output_names: Optional[Sequence[str]] = None,
               dynamic_axes: Optional[Dict] = None,
               verbose: bool = False):

        model = torch.load(pth_path)
        dummy_input = torch.randn(input_shape,device='cuda')
        torch.onnx.export(model, dummy_input, ir_path, verbose=True, input_names=input_names, output_names=output_names)

    def torch2torchscript(self, pth_path: str,
               input_shape: Sequence[int],
               ir_path:str):
        
        model = torch.load(pth_path)
        dummy_input = torch.randn(input_shape)
        mod = torch.jit.trace(model, dummy_input)
        mod.save(ir_path) 

    def torch2wts(self, pth_path: str,
                input_shape: Sequence[int],
                ir_path:str,
                cuda_id_str = "cuda:0",
               ):
        net = torch.load(pth_path)
        net = net.to(cuda_id_str)
        net = net.eval()
        print('model: ', net)
        #print('state dict: ', net.state_dict().keys())
        dummy_input = torch.randn(input_shape)
        tmp = dummy_input.to(cuda_id_str)
        print('input: ', tmp)
        out = net(tmp)

        print('output:', out)
        print(dummy_input[0].shape)
        summary(net, dummy_input[0].shape)
        #return
        f = open(ir_path, 'w')
        f.write("{}\n".format(len(net.state_dict().keys())))
        for k,v in net.state_dict().items():
            print('key: ', k)
            print('value: ', v.shape)
            vr = v.reshape(-1).cpu().numpy()
            f.write("{} {}".format(k, len(vr)))
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")