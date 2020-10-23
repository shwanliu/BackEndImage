import torch
from .models import FANet
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings("ignore")
# from torchvision import transforms
import torchvision.transforms as T
device = torch.device('cuda')

class getFaceAttr():
    def __init__(self, modelPath, value, classes):
        self.transform=T.Compose([
                            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                            T.Resize((224, 224)),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
        self.modelPath = modelPath
        self.value = value
        self.classes = classes
    def prediect(self,img_path):
        net = FANet(40)
        net.load_state_dict(torch.load(self.modelPath, map_location=torch.device('cpu'))['model'])
        torch.no_grad()
        net=net.eval()
        img=Image.open(img_path)
        draw_table = ImageDraw.Draw(im=img)
        # img.show()
        img_=self.transform(img).unsqueeze(0)
        #img_ = img.to(device)
        outputs = net(img_)
        zero = torch.zeros_like(outputs.data)
        one = torch.ones_like(outputs.data)
        predicted = torch.where(outputs.data > self.value, one, zero)
        pred = dict()
        # print("==="*15)
        for i in range(len(self.classes)):
            if outputs[0][i].item()>=self.value:
                pred[self.classes[i]]='%.2f'%outputs[0][i].item()
                score = float('%.4f' %outputs[0][i].item())
                # print("属性："+self.classes[i]+"===》score："+str(score))
        # print("==="*15)
        return pred
    # draw_table.text((100,100),"11",direction=None)
    # img.show()
    # _, predicted = torch.max(outputs, 1)
    # print(predicted)
if __name__ == '__main__':
    f = open('/Users/liuxiaoying/workplace/End-AIImage/class.txt')
    className = []
    lines = f.readlines()
    for line in lines:
        className.append( line.strip().split("：")[1])
    print(className)
    #modelPath = "/home/shawnliu/workPlace/face_attr/checkpoint/net_6.pkl"
    modelPath = "/Users/liuxiaoying/workplace/End-AIImage/faceAttr/checkpoints/epoch_60FANet.pth"
    # modelPath = "checkpoint/epoch10FANet.pth"
    faceAttr = getFaceAttr('/Users/liuxiaoying/workplace/End-AIImage/testliu2.jpg',modelPath,0.6,className)
    faceAttr.prediect()
    # prediect('testliu2.jpg',modelPath,0.6,className)
