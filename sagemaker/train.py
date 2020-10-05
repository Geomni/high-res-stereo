from __future__ import print_function

import argparse
import random
import time
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.functional import smooth_l1_loss
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from models import hsm
from utils import logger

from dataloader.train_dataloader import get_training_dataloader

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='HSM-Net')
    # region Data and Model Params
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'],
                        help='directory to save model checkpoints to')
    parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--train_folder', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--load_checkpoint', type=str, default=os.environ.get('SM_HP_LOAD_CHECKPOINT', None))
    parser.add_argument('--log_name', default=os.environ.get('SM_HP_LOG_NAME', 'out_log'), help='log name')
    # endregion

    # region Hyperparams
    parser.add_argument('--max_disp', type=int, default=os.environ.get('SM_HP_MAX_DISP', 385), help='maxium disparity')
    parser.add_argument('--epochs', type=int, default=os.environ.get('SM_HP_EPOCHS', 10),
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=os.environ.get('SM_HP_BATCH_SIZE', 2),
                        help='samples per batch')
    parser.add_argument('--seed', type=int, default=os.environ.get('SM_HP_SEED', 1), metavar='S',
                        help='random seed (default: 1)')
    # endregion

    args = parser.parse_args()
    return args


def initialize_model(train_args):
    print('Initializing model')
    torch.manual_seed(train_args.seed)

    model = hsm(train_args.max_disp, clean=False, level=1)
    model = nn.DataParallel(model)
    model.cuda()
    if train_args.load_checkpoint is not None:
        pretrained_dict = torch.load(train_args.loadmodel)
        pretrained_dict['state_dict'] = {k: v for k, v in pretrained_dict['state_dict'].items() if ('disp' not in k)}
        model.load_state_dict(pretrained_dict['state_dict'], strict=False)
    return model


def _initialize_dataset(train_args):
    return get_training_dataloader(train_args.max_disp, train_args.train_folder)


def main():
    train_args = parse_args()
    model = initialize_model(train_args)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    data_inuse = _initialize_dataset(train_args)
    train_img_loader = torch.utils.data.DataLoader(data_inuse, batch_size=train_args.batch_size,
                                                   shuffle=True, num_workers=train_args.batch_size,
                                                   drop_last=True, worker_init_fn=_init_fn)

    print('%d batches per epoch' % (len(data_inuse) // train_args.batch_size))

    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
    torch.manual_seed(train_args.seed)
    torch.cuda.manual_seed(train_args.seed)

    log = logger.Logger(train_args.output_data_dir, name=train_args.log_name)
    total_iters = 0

    for epoch in range(1, train_args.epochs + 1):
        total_train_loss = 0
        _adjust_learning_rate(optimizer, epoch, train_args)

        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(train_img_loader):
            start_time = time.time()
            loss, vis = _train(model, optimizer, imgL_crop, imgR_crop, disp_crop_L, train_args)
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
                save_filename = '{}/{}/finetune_{}.tar'.format(train_args.model_dir,
                                                               train_args.log_name,
                                                               total_iters)
                torch.save({'iters': total_iters,
                            'state_dict': model.state_dict(),
                            'train_loss': total_train_loss / len(train_img_loader),
                            }, save_filename)

        log.scalar_summary('train/loss', total_train_loss / len(train_img_loader), epoch)
        torch.cuda.empty_cache()


def _init_fn(worker_id):
    np.random.seed()
    random.seed()


def _loss_fn(output, disp_gt, mask):
    loss = (64. / 85) * smooth_l1_loss(output[0][mask], disp_gt[mask], size_average=True) + \
           (16. / 85) * smooth_l1_loss(output[1][mask], disp_gt[mask], size_average=True) + \
           (4. / 85) * smooth_l1_loss(output[2][mask], disp_gt[mask], size_average=True) + \
           (1. / 85) * smooth_l1_loss(output[3][mask], disp_gt[mask], size_average=True)
    return loss


def _train(model, optimizer, img_left, img_right, disp_left, train_args):
    model.train()
    img_left = Variable(torch.FloatTensor(img_left))
    img_right = Variable(torch.FloatTensor(img_right))
    disp_left = Variable(torch.FloatTensor(disp_left))

    img_left, img_right, disp_left = img_left.cuda(), img_right.cuda(), disp_left.cuda()

    # ---------
    mask = (disp_left > 0) & (disp_left < train_args.max_disp)
    mask.detach_()
    # ----

    optimizer.zero_grad()
    stacked, entropy = model(img_left, img_right)
    loss = _loss_fn(stacked, disp_left, mask)
    loss.backward()
    optimizer.step()
    vis = {'output3': stacked[0].detach().cpu().numpy(),
           'output4': stacked[1].detach().cpu().numpy(),
           'output5': stacked[2].detach().cpu().numpy(),
           'output6': stacked[3].detach().cpu().numpy(),
           'entropy': entropy.detach().cpu().numpy()}
    lossvalue = loss.data
    del stacked
    del loss
    return lossvalue, vis


def _adjust_learning_rate(optimizer, epoch, train_args):
    if epoch <= train_args.epochs - 1:
        lr = 1e-3
    else:
        lr = 1e-4
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
