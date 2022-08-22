import torch
import torch.nn as nn
import math


class TripletLoss_Siamese(nn.Module):
    def __init__(self, margin):
        super(TripletLoss_Siamese, self).__init__()
        self.margin = margin
        
    def forward(self, score_ap, score_an):
        return torch.clamp(torch.exp(1-score_ap) - torch.exp(1-score_an) + self.margin, 0).sum()

class MetrciSoftmaxLoss_Siamese(nn.Module):
    def __init__(self):
        super(MetrciSoftmaxLoss_Siamese, self).__init__()

    def forward(self, score_ap, score_an):

        return -torch.log(torch.exp(1 - score_an) / (torch.exp(1 - score_an) + torch.exp(1 - score_ap))).sum()

def hard_samples_mining(score_ap, score_an, margin):
    dis = torch.exp(1-score_ap) - torch.exp(1-score_an)
    idx = dis + margin > 0
    return idx


class MetricLoss_Siamese(nn.Module):
    def __init__(self, margin= 1, l=1.0):
        super(MetricLoss_Siamese, self).__init__()

        self.l = l
        self.margin = margin
        self.trip = TripletLoss_Siamese(margin)
        self.soft = MetrciSoftmaxLoss_Siamese()

    def forward(self, score_ap, score_an):
        loss_trip = self.trip(score_ap, score_an)
        loss_soft = self.soft(score_ap, score_an)

        return loss_trip + self.l * loss_soft

class MetricLoss_Siamese_1(nn.Module):
    def __init__(self, margin=1, l=1.0):
        super(MetricLoss_Siamese_1, self).__init__()

        self.l = l
        self.margin = margin
        self.trip = TripletLoss_Siamese(margin)
        self.soft = MetrciSoftmaxLoss_Siamese()

    def forward(self, score_ap, score_an):

        with torch.no_grad():
            idx = hard_samples_mining(score_ap, score_an, self.margin)
        loss_trip = self.trip(score_ap, score_an)
        loss_soft = self.soft(score_ap, score_an)

        return self.trip(score_ap[idx], score_an[idx]) + self.l * self.soft(score_ap[idx], score_an[idx])

