import torch
import csv

def val_score_siamese(score_ap, score_an):
    score = torch.exp(score_ap) / (torch.exp(score_ap) + torch.exp(score_an))
    return score

def write_csv(results, file_name):

    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)