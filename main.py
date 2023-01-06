'''
Copyright (c) 2023-present SMU PRALB
MIT license
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import time
import shutil
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from data.dataloader import get_dataloader
import models
from etc.help_function import *
from etc.utils import *
from etc.Visualize_video import ExportVideo
from etc.flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default='./config.json', help='JSON file for configuration')

    args = parser.parse_args()
    ############### setting framework ##########################################
    with open(args.config) as fin:
        config = json.load(fin)
    train_config = config['train_config']
    data_config = config['data_config']
    args.optim = train_config["optim"]
    args.lrsch = train_config["lrsch"]
    model_name = train_config["Model"]
    args.weight_decay = train_config["weight_decay"]
    others= args.weight_decay*0.01
    save_dir = train_config['save_dir']

    if not os.path.isdir(train_config['save_dir']):
        os.mkdir(train_config['save_dir'])
        
    shutil.copyfile("./config.json", save_dir + "/config.json")
    model = models.__dict__[model_name](num_classes=train_config["num_classes"])

    # print(train_config["num_classes"])
    batch = torch.FloatTensor(1, 3, data_config["w"], data_config["h"])
    model_eval = add_flops_counting_methods(model)
    model_eval.eval().start_flops_count()
    out = model_eval(batch)
    N_flop = model.compute_average_flops_cost()
    total_paramters = netParams(model)

    #################### common model setting and opt setting  #######################################
    start_epoch = 0
    Max_val_iou = 0.0
    Max_name = ''

    if train_config["resume"]:
        if os.path.isfile(train_config["resume"]):
            print("=> loading checkpoint '{}'".format(train_config["resume"]))
            checkpoint = torch.load(train_config["resume"])
            start_epoch = checkpoint['epoch']
            args.lr = checkpoint['lr']
            Max_name =  checkpoint['Max_name']
            Max_val_iou =  checkpoint['Max_val_iou']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(train_config["resume"], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(train_config["resume"]))

    use_cuda = torch.cuda.is_available()
    num_gpu = torch.cuda.device_count()

    if use_cuda:
        print("Use gpu : %d" % torch.cuda.device_count())
        if num_gpu > 1:
            model = torch.nn.DataParallel(model)
            print("make DataParallel")
        model = model.cuda()
        print("Done")

    ################################### stage Enc setting ##############################################
    trainLoader, valLoader = get_dataloader(data_config)
    criteria = nn.CrossEntropyLoss().cuda()

    if num_gpu > 0:
        criteria = criteria.cuda()

    if args.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), train_config['learning_rate'], \
            (0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), train_config["learning_rate"], \
            momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    elif args.optim == "RMS":
        optimizer = torch.optim.RMSprop(model.parameters(), train_config["learning_rate"], \
            alpha=0.9, momentum=0.9, eps=1, weight_decay=args.weight_decay)

    init_lr = train_config["learning_rate"]

    if args.lrsch == "multistep":
        decay = list(range(20, train_config["epochs"], 20))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay, gamma=0.5)
    elif args.lrsch == "step":
        step = train_config["epochs"] // 3
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.5)
    elif args.lrsch == "poly":
        lambda1 = lambda epoch: pow((1 - ((epoch - 1) / train_config["epochs"])), 0.9)  ## scheduler 2
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)  ## scheduler 2
    elif args.lrsch == "warmpoly":
        scheduler = WarmupPoly(init_lr=init_lr, total_ep=train_config["epochs"],
                            warmup_ratio=0.05, poly_pow=0.90)

    print("init_lr: " + str(train_config["learning_rate"]) + "   batch_size : " + str(data_config["batch_size"]) + "\n" +
        args.lrsch + " sch use weight and class " + str(train_config["num_classes"]))

    print('Flops:  {}'.format(flops_to_string(N_flop)))
    print('Params: ' + get_model_parameters_number(model))
    print('Output shape: [expression]{}'.format(list(out["expression"].shape)))
    print(total_paramters)

    ################################ start Enc train ##########################################
    
    min_loss = 999999
    lr = init_lr

    print("========== Stage 1 TRAINING ===========")
    for epoch in range(start_epoch, train_config["epochs"]):

        print('Train[%d/%d]\t' % (epoch+1, int(train_config["epochs"])) + "Learning rate: " + str(lr))
        lossTr = train(num_gpu, trainLoader, model, criteria, optimizer, epoch, train_config["epochs"])
        lossVal, acc, RMSE_valence, RMSE_arousal = validation(num_gpu, valLoader, model, criteria)
        print('[LossTr]: %.3f [LossVal]: %.3f [ACC]: %.2f [V_RMSE]: %.4f [A_RMSE]: %.4f'   % (lossTr, lossVal, acc, RMSE_valence, RMSE_arousal))

        if args.lrsch == "poly":
            scheduler.step(epoch)  ## scheduler 2
        elif args.lrsch == "warmpoly":
            curr_lr = scheduler.get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr
        else:
            scheduler.step()

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        if num_gpu > 1:
            this_state_dict = model.module.state_dict()
        else:
            this_state_dict = model.state_dict()

        if lossVal < min_loss and epoch > 0:
            min_loss = lossVal
            model_file_name = save_dir + f'/checkpoint_{epoch:03d}[L]{lossVal:.3f}[ACC]{acc:.2f}[V]{RMSE_valence:.3f}[A]{RMSE_arousal:.3f}.pth.tar'
            save_checkpoint({
                'epoch': epoch, 'arch': str(model),
                'state_dict': this_state_dict,
                'optimizer': optimizer.state_dict(),
                'lossTr': lossTr, 'lossVal': lossVal,
                'lr': lr,
                'ACC': acc, 'RMSE_valence': RMSE_valence,  'RMSE_arousal': RMSE_arousal
            }, model_file_name)

        print("Epoch : " + str(epoch+1) + ' Details')
        print("Epoch No.: %d\n" % (epoch+1))

        save_checkpoint({
            'epoch': epoch, 'arch': str(model),
            'state_dict': this_state_dict,
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr, 'lossVal': lossVal,
            'lr': lr,
            'ACC': acc, 'RMSE_valence': RMSE_valence,  'RMSE_arousal': RMSE_arousal
        }, save_dir + '/checkpoint.pth.tar')

    print("========== TRAINING FINISHED ===========")






