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

from model.deeplab import Res_Deeplab
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d, BCEWithLogitsLoss2d
from dataset.voc_dataset import VOCDataSet, VOCGTDataSet

import matplotlib.pyplot as plt
import random
import timeit
start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 10
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './dataset/VOC2012'
DATA_LIST_PATH = './dataset/voc_list/train_aug.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '321,321'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 20000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 20000
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
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv-pred", type=float, default=LAMBDA_ADV_PRED,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-semi", type=float, default=LAMBDA_SEMI,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--lambda-semi-adv", type=float, default=LAMBDA_SEMI_ADV,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--mask-T", type=float, default=MASK_T,
                        help="mask T for semi adversarial training.")
    parser.add_argument("--semi-start", type=int, default=SEMI_START,
                        help="start semi learning after # iterations")
    parser.add_argument("--semi-start-adv", type=int, default=SEMI_START_ADV,
                        help="start semi learning after # iterations")
    parser.add_argument("--discr-split", type=int)
    parser.add_argument("--D-remain", type=bool, default=D_REMAIN,
                        help="Whether to train D with unlabeled data")
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
    parser.add_argument("--restore-from-D", type=str, default=None,
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
    parser.add_argument("--gpu2", type=int, default=1)
    return parser.parse_args()

args = get_arguments()

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10


def one_hot(label):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)


def make_D_label(label, ignore_mask, gpu):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    D_label = np.ones(ignore_mask.shape)*label
    D_label[ignore_mask] = 255
    D_label = Variable(torch.FloatTensor(D_label)).cuda(gpu)

    return D_label


def main():
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
    gpu = args.gpu
    np.random.seed(args.random_seed)

    # create network
    model = Res_Deeplab(num_classes=args.num_classes)

    # load pretrained parameters
    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)

    # only copy the params that exist in current model (caffe-like)
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        print(name)
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
            print('copy {}'.format(name))
    model.load_state_dict(new_params)
    model.train()
    model.cuda(args.gpu)

    # init D
    model_D = FCDiscriminator(num_classes=args.num_classes)
    if args.restore_from_D is not None:
        model_D.load_state_dict(torch.load(args.restore_from_D))
    model_D.train()
    model_D.cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    # load dataset
    train_dataset = VOCDataSet(args.data_dir, args.data_list, crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

    train_dataset_size = len(train_dataset)

    train_gt_dataset = VOCGTDataSet(args.data_dir, args.data_list, crop_size=input_size,
                       scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

    if args.partial_data is None:
        trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)

        trainloader_gt = data.DataLoader(train_gt_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)
    else:
        #sample partial data
        partial_size = int(args.partial_data * train_dataset_size)

        if args.partial_id is not None:
            train_ids = pickle.load(open(args.partial_id))
            print('loading train ids from {}'.format(args.partial_id))
        else:
            train_ids = np.arange(train_dataset_size)
            np.random.shuffle(train_ids)

        pickle.dump(train_ids, open(osp.join(args.snapshot_dir, 'train_id.pkl'), 'wb'))

        # labeled data
        train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
        train_gt_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])

        trainloader = data.DataLoader(train_dataset,
                                      batch_size=args.batch_size, sampler=train_sampler, num_workers=3, pin_memory=True) 
        trainloader_gt = data.DataLoader(train_gt_dataset,
                                         batch_size=args.batch_size, sampler=train_gt_sampler, num_workers=3, pin_memory=True)
  
        # unlabeled data
        train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
        trainloader_remain = data.DataLoader(train_dataset,
                                             batch_size=args.batch_size, sampler=train_remain_sampler, num_workers=3, pin_memory=True)
        trainloader_remain_iter = enumerate(trainloader_remain)


    trainloader_iter = enumerate(trainloader)
    trainloader_gt_iter = enumerate(trainloader_gt)
    # implement model.optim_parameters(args) to handle different models' lr setting

    # optimizer for segmentation network
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D.zero_grad()

    # loss/bilinear upsampling
    bce_loss = BCEWithLogitsLoss2d()
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

    # labels for adversarial training
    pred_label = 0
    gt_label = 1

    for i_iter in range(args.num_steps):
        loss_seg_value = 0
        loss_adv_pred_value = 0
        
        loss_D_value = 0
        loss_D_ul_value = 0
        
        loss_semi_value = 0
        loss_semi_adv_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        # creating 2nd discriminator as a copy of the 1st one
        if i_iter == args.discr_split:
            model_D_ul = FCDiscriminator(num_classes=args.num_classes)    
            model_D_ul.load_state_dict(net_D.state_dict())         
            model_D_ul.train()
            model_D_ul.cuda(args.gpu)

            optimizer_D_ul = optim.Adam(model_D_ul.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
        
        # start training 2nd discriminator after specified number of steps        
        if i_iter >= args.discr_split:
            optimizer_D_ul.zero_grad()
            adjust_learning_rate_D(optimizer_D_ul, i_iter)

        for sub_i in range(args.iter_size):
        # train Segmentation

            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False
 
            # don't accumulate grads in D_ul, in case split has already been made
            if i_iter >= args.discr_split:
                for param in model_D_ul.parameters():
                    param.requires_grad = False

            # do semi-supervised training first
            if args.lambda_semi_adv > 0  and i_iter >= args.semi_start_adv :
                try:
                    _, batch = trainloader_remain_iter.next()
                except:
                    trainloader_remain_iter = enumerate(trainloader_remain)
                    _, batch = trainloader_remain_iter.next()

                # only access to img
                images, _, _, _ = batch
                images = Variable(images).cuda(args.gpu)

                pred = interp(model(images))
                pred_remain = pred.detach()
                
                # choose discriminator depending on the iteration
                if i_iter >= args.discr_split:
                    D_out = interp(model_D_ul(F.softmax(pred)))
                else:
                    D_out = interp(model_D(F.softmax(pred)))
                
                D_out_sigmoid = F.sigmoid(D_out).data.cpu().numpy().squeeze(axis=1)
                ignore_mask_remain = np.zeros(D_out_sigmoid.shape).astype(np.bool)

                # adversarial loss
                loss_semi_adv = args.lambda_semi_adv * bce_loss(D_out, make_D_label(gt_label, ignore_mask_remain, args.gpu))
                loss_semi_adv = loss_semi_adv / args.iter_size

                # true loss value without multiplier
                loss_semi_adv_value += loss_semi_adv.data.cpu().numpy() / args.lambda_semi_adv
                loss_semi_adv.backward()

            else:

                loss_semi = None
                loss_semi_adv = None


            # train with labeled images
            try:
                _, batch = trainloader_iter.next()
            except:
                trainloader_iter = enumerate(trainloader)
                _, batch = trainloader_iter.next()

            images, labels, _, _ = batch
            images = Variable(images).cuda(args.gpu)
            ignore_mask = (labels.numpy() == 255)

            pred = interp(model(images))
            D_out = interp(model_D(F.softmax(pred)))

            # computing loss
            loss_seg = loss_calc(pred, labels, args.gpu)
            loss_adv_pred = bce_loss(D_out, make_D_label(gt_label, ignore_mask, args.gpu))
            loss = loss_seg + args.lambda_adv_pred * loss_adv_pred

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value += loss_seg.data.cpu().numpy() / args.iter_size
            loss_adv_pred_value += loss_adv_pred.data.cpu().numpy() / args.iter_size


            # train D and D_ul

            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            if i_iter >= args.discr_split:
                for param in model_D_ul.parameters():
                    param.requires_grad = True
            
            # train D with pred
            pred = pred.detach()

            # before split, traing D with both labeled and unlabeled
            if args.D_remain and i_iter < args.discr_split and (args.lambda_semi > 0 or args.lambda_semi_adv > 0):
                pred = torch.cat((pred, pred_remain), 0)
                ignore_mask = np.concatenate((ignore_mask,ignore_mask_remain), axis = 0)

            D_out = interp(model_D(F.softmax(pred)))
            loss_D = bce_loss(D_out, make_D_label(pred_label, ignore_mask, args.gpu))
            loss_D = loss_D / args.iter_size / 2
            loss_D.backward()
            loss_D_value += loss_D.data.cpu().numpy()

            # train D_ul with pred on unlabeled
            if i_iter >= args.discr_split and (args.lambda_semi > 0 or args.lambda_semi_adv > 0):
                D_ul_out = interp(model_D_ul(F.softmax(pred_remain)))
                loss_D_ul = bce_loss(D_ul_out, make_D_label(pred_label, ignore_mask_remain, args.gpu))
                loss_D_ul = loss_D_ul / args.iter_size / 2
                loss_D_ul.backward()
                loss_D_ul_value += loss_D_ul.data.cpu().numpy()

            # get gt labels
            try:
                _, batch = trainloader_gt_iter.next()
            except:
                trainloader_gt_iter = enumerate(trainloader_gt)
                _, batch = trainloader_gt_iter.next()

            images_gt, labels_gt, _, _ = batch
            images_gt = Variable(images_gt).cuda(args.gpu)
            with torch.no_grad():
                pred_l = interp(model(images_gt))            

            # train D with gt
            D_gt_v = Variable(one_hot(labels_gt)).cuda(args.gpu)
            ignore_mask_gt = (labels_gt.numpy() == 255)

            D_out = interp(model_D(D_gt_v))
            loss_D = bce_loss(D_out, make_D_label(gt_label, ignore_mask_gt, args.gpu))
            loss_D = loss_D / args.iter_size / 2
            loss_D.backward()
            loss_D_value += loss_D.data.cpu().numpy()

            # train D_ul with pseudo_gt (gt are substituted for pred)
            if i_iter >= args.discr_split:
                D_ul_out = interp(model_D_ul(F.softmax(pred_l)))
                loss_D_ul = bce_loss(D_ul_out, make_D_label(gt_label, ignore_mask_gt, args.gpu))
                loss_D_ul = loss_D_ul / args.iter_size / 2
                loss_D_ul.backward()
                loss_D_ul_value += loss_D_ul.data.cpu().numpy()

        optimizer.step()
        optimizer_D.step()
        if i_iter >= args.discr_split:
            optimizer_D_ul.step()       
 

        print('exp = {}'.format(args.snapshot_dir))
        print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_p = {3:.3f}, loss_D = {4:.3f}, loss_D_ul={5:.3f}, loss_semi = {6:.3f}, loss_semi_adv = {7:.3f}'.format(i_iter, args.num_steps, loss_seg_value, loss_adv_pred_value, loss_D_value, loss_D_ul_value, loss_semi_value, loss_semi_adv_value))
        

        if i_iter >= args.num_steps-1:
            print ('save model ...')
            torch.save(net.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(args.num_steps) + '_' + str(args.random_seed) + '.pth'))
            torch.save(net_D.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(args.num_steps) + '_' + str(args.random_seed) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print ('taking snapshot ...')
            torch.save(net.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(i_iter) + '_' + str(args.random_seed) + '.pth'))
            torch.save(net_D.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+str(i_iter) + '_' + str(args.random_seed) + '_D.pth'))

    end = timeit.default_timer()
    print (end-start,'seconds')


if __name__ == '__main__':
    main()
