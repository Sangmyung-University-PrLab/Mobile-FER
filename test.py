'''
ExtPortraitSeg
Copyright (c) 2019-present NAVER Corp.
MIT license
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import time
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from data.dataloader import get_dataloader, get_test_loader
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
    args.weight_decay = train_config["weight_decay"]
    others= args.weight_decay*0.01
    save_dir = train_config['save_dir']

    if not os.path.isdir(train_config['save_dir']):
        os.mkdir(train_config['save_dir'])

    model = models.__dict__["mobile_former_151m"](num_classes=train_config["num_classes"])

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

    if train_config["test_params"]:
        if os.path.isfile(train_config["test_params"]):
            model.load_state_dict(torch.load(train_config["test_params"]))
        else:
            print("=> no checkpoint found at '{}'".format(train_config["test_params"]))

    use_cuda = torch.cuda.is_available()
    num_gpu = torch.cuda.device_count()

    if use_cuda:
        print("Use gpu : %d" % torch.cuda.device_count())
        if num_gpu > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        print("Done")

    ################################### stage Enc setting ##############################################
    # trainLoader, valLoader = get_dataloader(data_config)
    testLoader = get_test_loader(data_config)

    ################################ start Enc train ##########################################
    print("\n============ Evaluation TEST ==============")
    # switch to evaluation mode
    model.eval()

    correct = 0
    total = 0
    valence_se = 0
    arousal_se = 0

    for i, (input, label, _) in enumerate(testLoader):
        if num_gpu > 0:
            input = input.cuda()

            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                exp_label = label[:,0].type(torch.LongTensor)
                va_label = label[:,1]
                ar_label = label[:,2]

                exp_label = exp_label.cuda()
                va_label = va_label.cuda()
                ar_label = ar_label.cuda()

                # run the mdoel
                output = model(input_var)
                _, expression = torch.max(output["expression"], 1)
                total += exp_label.size(0)
                correct += (expression == exp_label).sum().item()
                
                valence = output["valence"]
                arousal = output["arousal"]

                valence = np.squeeze(valence.cpu().numpy())
                arousal = np.squeeze(arousal.cpu().numpy())

                va_label = np.squeeze(va_label.cpu().numpy())
                ar_label = np.squeeze(ar_label.cpu().numpy())

                valence_se += np.sum((valence - va_label)**2)
                arousal_se += np.sum((arousal - ar_label)**2)

    print(f'Accuracy of MODEL predictions: {100 * correct / total:0.2f} %')
    print(f'RMSE of VALENCE predictions: {np.sqrt(valence_se/total):0.4f} %')
    print(f'RMSE of AROUSAL predictions: {np.sqrt(arousal_se/total):0.4f} %')
    print("==========================================")






