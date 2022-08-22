from torchvision.models import  resnet50,  densenet121, resnext101_32x8d
import torch.nn as nn
import torch as t
import torch.nn.functional as F


'''

TL+FS

'''

class ResNet50_Siamese(nn.Module):
    def __init__(self):
        super(ResNet50_Siamese, self).__init__()
        self.model = resnet50(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True)
        )

        for params in self.model.parameters():
            params.requires_grad = False

        self.linear_1 = nn.Sequential(
            nn.Linear(256, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.linear_2 = nn.Sequential(
            nn.Linear(2048*3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x, y, z):
        b, c = x.size()[:2]

        x1 = self.model(x).view(b, -1)
        y1 = self.model(y).view(b, -1)
        z1 = self.model(z).view(b, -1)

        x1 = self.linear_1(x1)
        y1 = self.linear_1(y1)
        z1 = self.linear_1(z1)

        fcx1_y1 = t.mul(y1, x1)
        fcx1y1_cat = t.cat((y1, fcx1_y1, x1), 1)

        fcy1_z1 = t.mul(y1, z1)
        fcy1z1_cat = t.cat((y1, fcy1_z1, z1), 1)

        output_1 = self.linear_2(fcx1y1_cat)
        output_2 = self.linear_2(fcy1z1_cat)

        output_1 = F.softmax(output_1, dim=1)
        output_2 = F.softmax(output_2, dim=1)

        return output_1[:, 1], output_2[:, 1]

class ResNeXt101_Siamese(nn.Module):
    def __init__(self):
        super(ResNeXt101_Siamese, self).__init__()
        self.model = resnext101_32x8d(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True)
        )

        for params in self.model.parameters():
            params.requires_grad = False

        self.linear_1 = nn.Sequential(
            nn.Linear(256, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.linear_2 = nn.Sequential(
            nn.Linear(2048*3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x, y, z):
        b, c = x.size()[:2]

        x1 = self.model(x).view(b, -1)
        y1 = self.model(y).view(b, -1)
        z1 = self.model(z).view(b, -1)

        x1 = self.linear_1(x1)
        y1 = self.linear_1(y1)
        z1 = self.linear_1(z1)

        fcx1_y1 = t.mul(y1, x1)
        fcx1y1_cat = t.cat((y1, fcx1_y1, x1), 1)

        fcy1_z1 = t.mul(y1, z1)
        fcy1z1_cat = t.cat((y1, fcy1_z1, z1), 1)


        output_1 = self.linear_2(fcx1y1_cat)
        output_2 = self.linear_2(fcy1z1_cat)

        output_1 = F.softmax(output_1, dim=1)
        output_2 = F.softmax(output_2, dim=1)

        return output_1[:, 1], output_2[:, 1]


class DenseNet121_Siamese(nn.Module):
    def __init__(self):
        super(DenseNet121_Siamese, self).__init__()
        self.model = densenet121(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True)
        )

        for params in self.model.parameters():
            params.requires_grad = False

        self.linear_1 = nn.Sequential(
            nn.Linear(256, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.linear_2 = nn.Sequential(
            nn.Linear(2048*3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x, y, z):
        b, c = x.size()[:2]

        x1 = self.model(x).view(b, -1)
        y1 = self.model(y).view(b, -1)
        z1 = self.model(z).view(b, -1)

        x1 = self.linear_1(x1)
        y1 = self.linear_1(y1)
        z1 = self.linear_1(z1)

        fcx1_y1 = t.mul(y1, x1)
        fcx1y1_cat = t.cat((y1, fcx1_y1, x1), 1)

        fcy1_z1 = t.mul(y1, z1)
        fcy1z1_cat = t.cat((y1, fcy1_z1, z1), 1)

        output_1 = self.linear_2(fcx1y1_cat)
        output_2 = self.linear_2(fcy1z1_cat)

        output_1 = F.softmax(output_1, dim=1)
        output_2 = F.softmax(output_2, dim=1)

        return output_1[:, 1], output_2[:, 1]
