from torch.utils import data
from torchvision.transforms import transforms as T
from PIL import Image


class Triplet_Dataset(data.Dataset):
    def __init__(self, img_1, img_2, img_3, train=True, test=False):
        super(Triplet_Dataset, self).__init__()

        self.train = train
        self.test = test

        imgs_1_len = len(img_1)
        imgs_2_len = len(img_2)
        imgs_3_len = len(img_3)

        if self.test:
            self.img_1 = img_1
            self.img_2 = img_2
            self.img_3 = img_3

        elif self.train:
            self.img_1 = img_1[:int(1*imgs_1_len)]
            self.img_2 = img_2[:int(1*imgs_2_len)]
            self.img_3 = img_3[:int(1*imgs_3_len)]

        else:
            self.img_1 = img_1[int(0.8*imgs_1_len):]
            self.img_2 = img_2[int(0.8*imgs_2_len):]
            self.img_3 = img_3[int(0.8*imgs_3_len):]

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
         
        if self.test: 
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                normalize
            ])
        elif self.train:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        img_path_1 = self.img_1[index]
        img_path_2 = self.img_2[index]
        img_path_3 = self.img_3[index]

        if self.test:
            label_1 = img_path_1.split('.')[-2].split('/')[-1]
            label_2 = img_path_2.split('.')[-2].split('/')[-1]
            label_3 = img_path_3.split('.')[-2].split('/')[-1]

            label = label_1

        else:
            label = 1

        data_1=  Image.open(img_path_1)
        data_2 = Image.open(img_path_2)
        data_3 = Image.open(img_path_3)

        data_1 = self.transform(data_1)
        data_2 = self.transform(data_2)
        data_3 = self.transform(data_3)

        return data_1, data_2, data_3, label

    def __len__(self):
        return len(self.img_1)


class Triplet_Dataset_Val(data.Dataset):
    def __init__(self, img_1, img_2, img_3):

        self.img_1 = img_1
        self.img_2 = img_2
        self.img_3 = img_3

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            normalize
        ])

    def __getitem__(self, index):
        img_path_1 = self.img_1[index]
        img_path_2 = self.img_2[index]
        img_path_3 = self.img_3[index]

        label = 1 if len(img_path_1.split('/')[-1].split('_')) == 3 else 0

        data_1 = Image.open(img_path_1)
        data_2 = Image.open(img_path_2)
        data_3 = Image.open(img_path_3)

        data_1 = self.transform(data_1)
        data_2 = self.transform(data_2)
        data_3 = self.transform(data_3)

        return data_1, data_2, data_3, label

    def __len__(self):
        return len(self.img_1)






