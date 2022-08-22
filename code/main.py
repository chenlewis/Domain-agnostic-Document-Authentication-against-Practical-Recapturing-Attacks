from config import opt
import torch
from tqdm import tqdm
from models import model as models
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import os
from data.dataset import Triplet_Dataset, Triplet_Dataset_Val
from loss.loss import MetricLoss_Siamese, MetricLoss_Siamese_1
from utils.utils import val_score_siamese, write_csv
import numpy as np


def train(**kwargs):
    opt._parse(kwargs)
    model = getattr(models, opt.model)()

    img_1 = [os.path.join(opt.train_data_root1, img) for img in os.listdir(opt.train_data_root1)]
    img_2 = [os.path.join(opt.train_data_root2, img) for img in os.listdir(opt.train_data_root2)]
    img_3 = [os.path.join(opt.train_data_root3, img) for img in os.listdir(opt.train_data_root3)]

    img_4 = [os.path.join(opt.val_data_root1, img) for img in os.listdir(opt.val_data_root1)]
    img_5 = [os.path.join(opt.val_data_root2, img) for img in os.listdir(opt.val_data_root2)]
    img_6 = [os.path.join(opt.val_data_root3, img) for img in os.listdir(opt.val_data_root3)]

    img_1 = sorted(img_1)
    img_2 = sorted(img_2)
    img_3 = sorted(img_3)

    img_4 = sorted(img_4)
    img_5 = sorted(img_5)
    img_6 = sorted(img_6)

    shuffle_data = True             
    if shuffle_data:
        c = list(zip(img_1, img_2, img_3))
        random.shuffle(c)
        img_1, img_2, img_3 = zip(*c)

    if shuffle_data:
        c = list(zip(img_4, img_5, img_6))
        random.shuffle(c)
        img_4, img_5, img_6 = zip(*c)

    train_data = Triplet_Dataset(img_1, img_2, img_3, train=True, test=False)
    train_dataloader = DataLoader(train_data, batch_size=opt.train_batch_size, shuffle=False, num_workers=opt.num_workers)

    val_data = Triplet_Dataset_Val(img_4, img_5, img_6)
    val_dataloader = DataLoader(val_data, batch_size=opt.train_batch_size, shuffle=False, num_workers=opt.num_workers)
    
    criterion = MetricLoss_Siamese_1()
    optimizer = optim.Adam(params=model.parameters(), lr=opt.lr)
    
    
    model.load_state_dict(torch.load(opt.load_train_model_path, map_location='cpu'), strict=False)
    model.to(opt.device)
    step = 0
    best_acc = 0
    for epoch in range(opt.max_epoch):
        running_loss = 0
        model.train()
        for anchors, positives, negatives, label in tqdm(train_dataloader):

            anchors = anchors.to(opt.device)
            positives = positives.to(opt.device)
            negatives = negatives.to(opt.device)

            '''
            Siamese-Network
            '''
            score_ap, score_an = model(positives, anchors, negatives)
            loss = criterion(score_ap, score_an)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 100000 == 0:
                print('-step: {} -loss: {}'.format(step, loss.item()))

            step += 1
        print('-epoch:{} -loss:{}'.format(epoch, running_loss))

        acc = Val(model, val_dataloader)
        print("-epoch:  {} accuracy: {}".format(epoch, acc))

        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "./temp-epoch{}".format(epoch))

        if opt.scheduler is not None:
            opt.scheduler.step()
            
            
def Val(model, dataloader):
    model.eval()
    correct = 0
    for detection, positives, negatives, label in tqdm(dataloader):

        detection = detection.to(opt.device)
        positives = positives.to(opt.device)
        negatives = negatives.to(opt.device)

        label = label.to(opt.device)

        score_ap, score_an = model(positives, detection, negatives)

        out = (score_ap-score_an).data.cpu().numpy()
        pred = np.where(out>0, 1, 0)

        num_correct = (pred == label.long().data.cpu().numpy()).sum().item()
        
        correct += num_correct

    acc = correct / len(dataloader.dataset)
    return acc


def test(**kwargs):
    opt._parse(kwargs)
    model = getattr(models, opt.model)()
    
    model.load_state_dict(torch.load(opt.load_test_model_path, map_location='cpu'))

    img_1 = [os.path.join(opt.test_data_root1, img) for img in os.listdir(opt.test_data_root1)]
    img_2 = [os.path.join(opt.test_data_root2, img) for img in os.listdir(opt.test_data_root2)]
    img_3 = [os.path.join(opt.test_data_root3, img) for img in os.listdir(opt.test_data_root3)]

    img_1 = sorted(img_1)
    img_2 = sorted(img_2)
    img_3 = sorted(img_3)

    shuffle_data = True
    if shuffle_data:
        c = list(zip(img_1, img_2, img_3))
        random.shuffle(c)
        img_1, img_2, img_3 = zip(*c)

    test_data = Triplet_Dataset(img_1, img_2, img_3, train=False, test=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.num_workers)


    model.to(opt.device)
    results = []
    model.eval()
    for Detection, Positive, Negative, label in tqdm(test_dataloader):
        detection = Detection.to(opt.device)
        positives = Positive.to(opt.device)
        negatives = Negative.to(opt.device)

        label = label
        score_1, score_2 = model(positives, detection, negatives)
        
        output = val_score_siamese(score_1, score_2).detach().tolist()
        batch_results = [(label_, output_) for label_, output_ in zip(label, output)]
        results += batch_results

    write_csv(results, opt.result_file)
    
    

if __name__ == '__main__':
    import fire
    fire.Fire()