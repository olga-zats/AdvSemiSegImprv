import argparse
import cv2
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
import pickle
from packaging import version

from utils.loss import BCEWithLogitsLoss2d
from dataset.voc_dataset_cl import VOCClsDataSet
from model.deeplab_class import Res_Deeplab

import matplotlib.pyplot as plt
import random
import timeit
start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 5
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './dataset/VOC2012'
DATA_LIST_PATH = './dataset/voc_list/train_aug.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '505, 505'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 20000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_ADV_PRED = 0.1

PARTIAL_DATA=0.5

SEMI_START=5000
LAMBDA_SEMI=0.1
MASK_T=0.2

LAMBDA_SEMI_ADV=0.001
SEMI_START_ADV=0
D_REMAIN=True


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--partial-data", type=float, default=PARTIAL_DATA,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--partial-id", type=str, default=None,
                        help="restore partial id list")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--snapshot-prefix", type=str, default='',
                        help="Identifies experiment")
    parser.add_argument("--seed", type=int, default=0,
                        help="Identifies sample")
    parser.add_argument("--mode", type=int, default=6,
                        help="Identifies sample")
    return parser.parse_args()

args = get_arguments()

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def make_cls_label(label, num_classes=21):
    cls_label = np.zeros((label.shape[0], num_classes - 1, 1, 1))
    for i in range(label.shape[0]):
        clsIDs = np.unique(label[i,:,:]).astype(np.uint8)
        for clsID in clsIDs:
            if clsID in [0, 255]:
                continue
            cls_label[i, clsID - 1, 0, 0] = 1.0

    cls_label = Variable(torch.FloatTensor(cls_label)).cuda(args.gpu)

    return cls_label

def main():

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
    gpu = args.gpu

    # create network
    model = Res_Deeplab(num_classes=args.num_classes, mode=args.mode)

    # load pretrained parameters
    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)

    # only copy the params that exist in current model (caffe-like)
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        print name
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
            print('copy {}'.format(name))
    model.load_state_dict(new_params)


    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    bce_loss = BCEWithLogitsLoss2d()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


    train_dataset = VOCClsDataSet(args.data_dir, args.data_list, crop_size=input_size,
                    scale=True, mirror=True, mean=IMG_MEAN)

    train_dataset_size = len(train_dataset)
    
    if args.partial_data is None:
        trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)
    else:
        #sample partial data
        partial_size = int(args.partial_data * train_dataset_size)

        if args.partial_id is not None:
            train_ids = pickle.load(open(args.partial_id))
            print('loading train ids from {}'.format(args.partial_id))
        else:
            train_ids = range(train_dataset_size)
            np.random.seed(args.seed)
            np.random.shuffle(train_ids)
            #print(train_ids)

        pickle.dump(train_ids, open(osp.join(args.snapshot_dir, 'train_id_seed_' + str(args.seed) + '_.pkl'), 'wb'))

        train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])

        trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, sampler=train_sampler, num_workers=3, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    # optimizer for segmentation network
    optimizer = optim.SGD(model.optim_parameters(args),
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    for i_iter in range(args.num_steps):
        loss_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        for sub_i in range(args.iter_size):
            # train with source

            try:
                _, batch = trainloader_iter.next()
            except:
                trainloader_iter = enumerate(trainloader)
                _, batch = trainloader_iter.next()

            images, cls_label, _, _ = batch
            images = Variable(images).cuda(args.gpu)
            cls_pred = model(images)

	    cls_label = Variable(torch.FloatTensor(cls_label)).cuda(args.gpu)
            loss = bce_loss(torch.unsqueeze(torch.unsqueeze(cls_pred, 2), 3), cls_label)

            # proper normalization
            loss = loss/args.iter_size
            loss.backward()
            loss_value += loss.data.cpu().numpy()/args.iter_size

        optimizer.step()

        print('iter = {0:8d}/{1:8d}, loss = {2:.3f}'.format(i_iter, args.num_steps, loss_value))

        if i_iter >= args.num_steps-1:
            print 'save model ...'
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'VOC_classifier_VOCcls_pd_' + str(args.partial_data) + '_seed_' + str(args.seed) + '_' +str(args.num_steps)+'.pth'))

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print 'taking snapshot ...'
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'VOC_classifier_VOCcls_pd_' + str(args.partial_data) + '_seed_' + str(args.seed) + '_' +str(i_iter)+'.pth'))

    end = timeit.default_timer()
    print end-start,'seconds'

if __name__ == '__main__':
    main()
