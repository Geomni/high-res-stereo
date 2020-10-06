from __future__ import print_function
from dataloader import listfiles as ls
from dataloader import MiddleburyLoader as DA
from dataloader import KITTIloader2012 as lk12
from dataloader import KITTIloader2015 as lk15
from dataloader import listsceneflow as lt
import pdb
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from models import hsm
from utils import logger
from utils.s3_sync import sync_open_dataset

torch.backends.cudnn.benchmark = True
MODEL = None
OPTIMIZER = None


def parse_args():
    parser = argparse.ArgumentParser(description='HSM-Net')
    parser.add_argument('--maxdisp', type=int, default=384, help='maxium disparity')
    parser.add_argument('--logname', default='logname', help='log name')
    parser.add_argument('--database', default='./data', help='data path')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batchsize', type=int, default=28, help='samples per batch')
    parser.add_argument('--loadmodel', default=None, help='weights path')
    parser.add_argument('--savemodel', default='./model', help='save path')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()
    return args


def load_model(input_args):
    global MODEL, OPTIMIZER
    torch.manual_seed(input_args.seed)
    MODEL = hsm(input_args.maxdisp, clean=False, level=1)
    MODEL = nn.DataParallel(MODEL)
    MODEL.cuda()

    # load model
    if input_args.loadmodel is not None:
        pretrained_dict = torch.load(input_args.loadmodel)
        pretrained_dict['state_dict'] = {k: v for k, v in pretrained_dict['state_dict'].items() if ('disp' not in k)}
        MODEL.load_state_dict(pretrained_dict['state_dict'], strict=False)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in MODEL.parameters()])))
    OPTIMIZER = optim.Adam(MODEL.parameters(), lr=0.1, betas=(0.9, 0.999))
    torch.manual_seed(input_args.seed)  # set again
    torch.cuda.manual_seed(input_args.seed)


def _init_fn(worker_id):
    np.random.seed()
    random.seed()


def init_dataloader(input_args):
    batch_size = input_args.batchsize
    scale_factor = input_args.maxdisp / 384.  # controls training resolution
    all_left_img, all_right_img, all_left_disp, all_right_disp = ls.dataloader(
        '%s/carla-highres/trainingF' % input_args.database)
    loader_carla = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, right_disparity=all_right_disp, rand_scale=[
                                    0.225, 0.6*scale_factor], rand_bright=[0.8, 1.2], order=2)

    all_left_img, all_right_img, all_left_disp, all_right_disp = ls.dataloader(
        '%s/mb-ex-training/trainingF' % input_args.database)  # mb-ex
    loader_mb = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, right_disparity=all_right_disp, rand_scale=[
        0.225, 0.6*scale_factor], rand_bright=[0.8, 1.2], order=0)

    all_left_img, all_right_img, all_left_disp, all_right_disp = lt.dataloader('%s/sceneflow/' % input_args.database)
    loader_scene = DA.myImageFloder(all_left_img, all_right_img, all_left_disp,
                                    right_disparity=all_right_disp, rand_scale=[0.9, 2.4*scale_factor], order=2)

    all_left_img, all_right_img, all_left_disp, _, _, _ = lk15.dataloader(
        '%s/kitti_scene/training/' % input_args.database, typ='train')  # change to trainval when finetuning on KITTI
    loader_kitti15 = DA.myImageFloder(all_left_img, all_right_img, all_left_disp,
                                      rand_scale=[0.9, 2.4*scale_factor], order=0)
    all_left_img, all_right_img, all_left_disp = lk12.dataloader('%s/data_stereo_flow/training/' % input_args.database)
    loader_kitti12 = DA.myImageFloder(all_left_img, all_right_img, all_left_disp,
                                      rand_scale=[0.9, 2.4*scale_factor], order=0)

    all_left_img, all_right_img, all_left_disp, _ = ls.dataloader('%s/eth3d/' % input_args.database)
    loader_eth3d = DA.myImageFloder(all_left_img, all_right_img, all_left_disp,
                                    rand_scale=[0.9, 2.4*scale_factor], order=0)

    data_inuse = torch.utils.data.ConcatDataset(
        [loader_carla]*40 + [loader_mb]*500 + [loader_scene] + [loader_kitti15] + [loader_kitti12]*80 + [loader_eth3d]*1000)
    TrainImgLoader = torch.utils.data.DataLoader(
        data_inuse, batch_size=batch_size, shuffle=True, num_workers=batch_size, drop_last=True, worker_init_fn=_init_fn)
    print('%d batches per epoch' % (len(data_inuse)//batch_size))
    return TrainImgLoader


def train(imgL, imgR, disp_L):
    MODEL.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    mask = (disp_true > 0) & (disp_true < args.maxdisp)
    mask.detach_()
    # ----

    OPTIMIZER.zero_grad()
    stacked, entropy = model(imgL, imgR)
    loss = (64./85)*F.smooth_l1_loss(stacked[0][mask], disp_true[mask], size_average=True) + \
           (16./85)*F.smooth_l1_loss(stacked[1][mask], disp_true[mask], size_average=True) + \
        (4./85)*F.smooth_l1_loss(stacked[2][mask], disp_true[mask], size_average=True) + \
        (1./85)*F.smooth_l1_loss(stacked[3][mask], disp_true[mask], size_average=True)
    loss.backward()
    OPTIMIZER.step()
    vis = {}
    vis['output3'] = stacked[0].detach().cpu().numpy()
    vis['output4'] = stacked[1].detach().cpu().numpy()
    vis['output5'] = stacked[2].detach().cpu().numpy()
    vis['output6'] = stacked[3].detach().cpu().numpy()
    vis['entropy'] = entropy.detach().cpu().numpy()
    lossvalue = loss.data

    del stacked
    del loss
    return lossvalue, vis


def adjust_learning_rate(optimizer, epoch, input_args):
    if epoch <= input_args.epochs - 1:
        lr = 1e-3
    else:
        lr = 1e-4
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    input_args = parse_args()
    sync_open_dataset(input_args.database)

    load_model(input_args)

    log = logger.Logger(input_args.savemodel, name=input_args.logname)
    total_iters = 0

    for epoch in range(1, input_args.epochs + 1):
        total_train_loss = 0
        adjust_learning_rate(OPTIMIZER, epoch, input_args)

        ## training ##
        train_img_loader = init_dataloader(input_args)
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(train_img_loader):
            start_time = time.time()
            loss, vis = train(imgL_crop, imgR_crop, disp_crop_L)
            print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
            total_train_loss += loss

            if total_iters % 10 == 0:
                log.scalar_summary('train/loss_batch', loss, total_iters)
            if total_iters % 100 == 0:
                log.image_summary('train/left', imgL_crop[0:1], total_iters)
                log.image_summary('train/right', imgR_crop[0:1], total_iters)
                log.image_summary('train/gt0', disp_crop_L[0:1], total_iters)
                log.image_summary('train/entropy', vis['entropy'][0:1], total_iters)
                log.histo_summary('train/disparity_hist', vis['output3'], total_iters)
                log.histo_summary('train/gt_hist', np.asarray(disp_crop_L), total_iters)
                log.image_summary('train/output3', vis['output3'][0:1], total_iters)
                log.image_summary('train/output4', vis['output4'][0:1], total_iters)
                log.image_summary('train/output5', vis['output5'][0:1], total_iters)
                log.image_summary('train/output6', vis['output6'][0:1], total_iters)

            total_iters += 1

            if (total_iters + 1) % 2000 == 0:
                # SAVE
                savefilename = input_args.savemodel+'/'+input_args.logname+'/finetune_'+str(total_iters)+'.tar'
                torch.save({
                    'iters': total_iters,
                    'state_dict': MODEL.state_dict(),
                    'train_loss': total_train_loss/len(train_img_loader),
                }, savefilename)

        log.scalar_summary('train/loss', total_train_loss/len(train_img_loader), epoch)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
