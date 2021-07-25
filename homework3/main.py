# David Fried
# 4/1/2021
# Peizhao's code adapted for the landmark dataset problem
import time

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import MSELoss

from model import LandmarkCNN
from dataloader import WFLWDataset, train_t, test_t
import parser
from math import sqrt


import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Landmarks")

    parser.add_argument('--seed', type=int, default=234)
    parser.add_argument('--device', type=str, default="cuda:0")

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval_interval', type=int, default=20)

    parser.add_argument('--train_file', type=str, default="WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt")
    parser.add_argument('--eval_file', type=str, default="WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt")
    parser.add_argument('--img_folder', type=str, default="./data")

    return parser.parse_args()



def eval(data_loader: DataLoader, model: LandmarkCNN, device: torch.device):
    model.eval()
    error = 0
    batches = 0
    loss = MSELoss()

    with torch.no_grad():
        for sample in data_loader:
            source = sample["image"]
            target = sample["landmarks"]
            source = source.to(device)
            target = target.to(device)

            landmarks = model(source)
            error += loss(landmarks, target)
            batches += 1
    rmse = sqrt(error / batches)
    print(f"RMSE: {rmse}")


def main():
    args = parse_args()

    device = torch.device(args.device)

    train_set = WFLWDataset(args.train_file, args.img_folder, train=True, transform=train_t)
    eval_set = WFLWDataset(args.eval_file, args.img_folder, train=True, transform=test_t)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size)

    model = LandmarkCNN(landmark_num=train_set.n_landmarks)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = MSELoss()

    for epoch in range(1, args.epoch + 1):
        model.train()

        for sample in train_loader:
            source = sample["image"]
            target = sample["landmarks"]
            source = source.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            y_pred = model(source)
            loss = loss_func(y_pred, target)

            loss.backward()
            optimizer.step()

            print(loss)

        print(f"{epoch=}")

        if epoch % args.eval_interval == 0:
            eval(eval_loader, model, device)


if __name__ == "__main__":
    main()
