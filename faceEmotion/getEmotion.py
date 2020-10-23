import torch
from .models import *
from PIL import Image, ImageDraw, ImageFont
import warnings
import torch.nn.functional as F 
import torchvision.transforms as transforms
warnings.filterwarnings("ignore")
import torchvision.transforms as T
device = torch.device('cuda')
transform=T.Compose([
        transforms.CenterCrop(42),
        transforms.Grayscale(3),
        transforms.ToTensor(),
                            ])

class getEmotion():
    def __init__(self, modelPath,classes):
        self.transform=T.Compose([
                            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                            T.Resize((224, 224)),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
        self.modelPath = modelPath
        self.classes = classes
    
    def prediect(self, img_path):
        net = ERNet(7)
        net.load_state_dict(torch.load(self.modelPath, map_location=torch.device('cpu'))['model'])
        torch.no_grad()
        net=net.eval()
        img=Image.open(img_path)
        draw_table = ImageDraw.Draw(im=img)
        img_=transform(img).unsqueeze(0)
        outputs = net(img_)

        predicted = F.softmax(outputs)

        _, idx = torch.max(predicted, 1)
        res = ""
        for (key,value) in self.classes.items():
            if value == idx:
                res = key 
        return res

if __name__ == '__main__':
    modelPath = "checkpoints/epoch_99ERNet.pth"
    labels = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprised': 6}
    img_path = "/Users/liuxiaoying/workplace/End-AIImage/testliu2.jpg"
    faceEmotion = getEmotion(modelPath, labels)
    emotion = faceEmotion.prediect(img_path)
    print(emotion)
