import torch
import torch.nn as nn
from etc.IOUEval import iouEval
import time
import os
import torchvision
import numpy as np
from etc.lovasz_losses import iou_binary, calcF1
import cv2
import datetime
from tqdm import tqdm

def mse_loss(pred, target):
    loss_func = nn.MSELoss()
    loss = loss_func(pred, target)
    return loss

def teacher_bound(num_gpu, criterion, expression, valence, arousal, label, distill):
    if num_gpu > 0:
        loss = criterion(expression, label[:,0].type(torch.LongTensor).cuda())
        valence_loss = mse_loss(valence, label[:,1])
        arousal_loss = mse_loss(arousal, label[:,2])

        kd_loss = criterion(expression, torch.argmax(distill[:,:-2], dim=1).type(torch.LongTensor).cuda())
        kd_valence_loss = mse_loss(valence, distill[:,-2])
        kd_arousal_loss = mse_loss(arousal, distill[:,-1])
    else:
        loss = criterion(expression, label[:,0].type(torch.LongTensor))
        valence_loss = mse_loss(valence, label[:,1])
        arousal_loss = mse_loss(arousal, label[:,2])

        kd_loss = criterion(expression, distill[:,:-2])
        kd_valence_loss = mse_loss(valence, distill[:,-2])
        kd_arousal_loss = mse_loss(arousal, distill[:,-1])

    if valence_loss + 0.5 > mse_loss(distill[:,-2], label[:, 1]) or arousal_loss + 0.5 > mse_loss(distill[:,-1], label[:, 2]):
        loss = 0.7 * (loss + 1.7 * valence_loss + 1.7 * arousal_loss) + \
            0.3 * (kd_loss + kd_valence_loss + kd_arousal_loss)
    else:
        loss = 0.7 * (loss + valence_loss + arousal_loss) + \
            0.3 * (kd_loss + kd_valence_loss + kd_arousal_loss)

    return loss

def kd_loss(num_gpu, criterion, expression, valence, arousal, label, distill):
    if num_gpu > 0:
        loss = criterion(expression, label[:,0].type(torch.LongTensor).cuda())
        valence_loss = mse_loss(valence, label[:,1])
        arousal_loss = mse_loss(arousal, label[:,2])

        kd_loss = criterion(expression, torch.argmax(distill[:,:-2], dim=1).type(torch.LongTensor).cuda())
        kd_valence_loss = mse_loss(valence, distill[:,-2])
        kd_arousal_loss = mse_loss(arousal, distill[:,-1])
    else:
        loss = criterion(expression, label[:,0].type(torch.LongTensor))
        valence_loss = mse_loss(valence, label[:,1])
        arousal_loss = mse_loss(arousal, label[:,2])

        kd_loss = criterion(expression, distill[:,:-2])
        kd_valence_loss = mse_loss(valence, distill[:,-2])
        kd_arousal_loss = mse_loss(arousal, distill[:,-1])

    loss = 0.7 * (loss + valence_loss + arousal_loss) + 0.3 * (kd_loss + kd_valence_loss + kd_arousal_loss)
    return loss
    

def validation(num_gpu, val_loader, model, criterion):
    '''
    :param val_loader: loaded for validation dataset
    :param model: model
    :param criterion: loss function
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu
    '''
    # switch to evaluation mode
    model.eval()

    epoch_loss = []

    correct = 0
    total = 0
    valence_se = 0
    arousal_se = 0

    for i, (input, label, distill) in enumerate(val_loader):
        if num_gpu > 0:
            input = input.cuda()

        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            exp_label = label[:,0].type(torch.LongTensor)
            va_label = label[:,1]
            ar_label = label[:,2]

            if num_gpu > 0:
                label = label.cuda()
                distill = distill.cuda()
                exp_label = exp_label.cuda()
                va_label = va_label.cuda()
                ar_label = ar_label.cuda()

            # run the mdoel
            output = model(input_var)
            expression = output["expression"]
            valence = output["valence"]
            arousal = output["arousal"]

            loss = teacher_bound(num_gpu, criterion, expression, valence, arousal, label, distill)
            epoch_loss.append(loss.item())
            
            _, expression = torch.max(output["expression"], 1)
            total += exp_label.size(0)
            correct += (expression == exp_label).sum().item()
            
            valence = np.squeeze(valence.cpu().numpy())
            arousal = np.squeeze(arousal.cpu().numpy())
            
            va_label = np.squeeze(va_label.cpu().numpy())
            ar_label = np.squeeze(ar_label.cpu().numpy())

            valence_se += np.sum((valence - va_label)**2)
            arousal_se += np.sum((arousal - ar_label)**2)

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)

    acc = 100 * correct / total
    RMSE_valence = np.sqrt(valence_se/total)
    RMSE_arousal = np.sqrt(arousal_se/total)

    return average_epoch_loss_val, acc, RMSE_valence, RMSE_arousal


def train(num_gpu, train_loader, model, criterion, optimizer, epoch, total_ep):
    '''
    :param train_loader: loaded for training dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    # switch to train mode
    model.train()
    epoch_loss = []
    for i, (input, label, distill) in enumerate(tqdm(train_loader)):

        if num_gpu > 0:
            input = input.cuda()
            label = label.cuda()
            distill = distill.cuda()

        input_var = torch.autograd.Variable(input)
        label = torch.autograd.Variable(label)
        distill = torch.autograd.Variable(distill)

        # run the mdoel
        output = model(input_var)
        expression = output["expression"]
        valence = output["valence"]
        arousal = output["arousal"]

        # set the grad to zero
        optimizer.zero_grad()

        loss = teacher_bound(num_gpu, criterion, expression, valence, arousal, label, distill)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    
    return average_epoch_loss_train


def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    '''
    helper function to save the checkpoint
    :param state: model state
    :param filenameCheckpoint: where to save the checkpoint
    :return: nothing
    '''
    torch.save(state, filenameCheckpoint)

def netParams(model):
    '''
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p
        # print(total_paramters)

    return total_paramters