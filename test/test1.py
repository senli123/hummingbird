import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import cv2
import numpy as np
def save_pth():
    print('cuda device count: ', torch.cuda.device_count())
    net = torchvision.models.alexnet(pretrained=True)
    #net.fc = nn.Linear(512, 2)
    net.eval()
    net = net.to('cuda:0')
    print(net)
    tmp = torch.ones(2, 3, 224, 224).to('cuda:0')
    out = net(tmp)
    print('alexnet out:', out.shape)
    torch.save(net, "alexnet.pth")
  
def pytorch_infer():
    path = "/workspace/lisen/tensorrt/my_tensorrt/img/dog.jpg"
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    mean_array = np.array([0.485, 0.456, 0.406], dtype= np.float32) * 255
    std_array = np.reciprocal(np.array([0.229, 0.224, 0.225], dtype= np.float32)* 255, dtype=np.float32)
    image_array = np.array(img, dtype=np.float32, order='C')
    image_array -= mean_array
    image_array *= std_array
    image_array = np.transpose(image_array, [2,0,1])
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.array(image_array, dtype=np.float32, order='C')
    net = torchvision.models.alexnet(pretrained=True)
    #net.fc = nn.Linear(512, 2)
    net.eval()
    net = net.to('cuda:0')
    print(net)
    tmp = torch.from_numpy(image_array).to('cuda:0')
    out = net(tmp)
    out = out.cpu()
    out = out.detach().numpy()
    top_ind = np.argsort(out[0])[-1:][::-1]
    print(top_ind[-1])

def check():
    path = "/workspace/lisen/tensorrt/my_tensorrt/model_zoo/pth/alexnet.pth"
    model = torch.load(path)
    net = torchvision.models.alexnet(pretrained=True)
    print(next(net.parameters()).device)  
if __name__ == '__main__':
    check()

