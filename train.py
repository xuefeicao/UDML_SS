# coding=utf-8
from __future__ import absolute_import, print_function
import time
import argparse
import os
import sys
import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)
import torch.utils.data
from torch.backends import cudnn
from torch.autograd import Variable
import glob
import models
import losses
from utils import FastRandomIdentitySampler, mkdir_if_missing, logging, display, create_fake_labels
from utils.serialization import save_checkpoint, load_checkpoint
from trainer import train
from utils import orth_reg
from evaluations import extract_features
from test import extract_recalls

from data import dataset
import numpy as np
import os.path as osp
import ast
cudnn.benchmark = True

use_gpu = True


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def main(args):
    # s_ = time.time()
    print(torch.cuda.get_device_properties(device=0).total_memory)
    torch.cuda.empty_cache()
    print(args)
    save_dir = args.save_dir
    mkdir_if_missing(save_dir)
    num_txt = len(glob.glob(save_dir + "/*.txt"))
    sys.stdout = logging.Logger(os.path.join(save_dir, "log_" + str(num_txt) + ".txt"))
    display(args)
    start = 0
   

    model = models.create(args.net, pretrained=args.pretrained, dim=args.dim, self_supervision_rot=args.self_supervision_rot)
    all_pretrained = glob.glob(save_dir + "/*.pth.tar")

    if (args.resume is None) or (len(all_pretrained) == 0):
        model_dict = model.state_dict()

    else:
        # resume model
        all_pretrained_epochs = sorted([int(x.split("/")[-1][6:-8]) for x in all_pretrained])
        args.resume = os.path.join(save_dir, "ckp_ep" + str(all_pretrained_epochs[-1]) + ".pth.tar")
        print('load model from {}'.format(args.resume))
        chk_pt = load_checkpoint(args.resume)
        weight = chk_pt['state_dict']
        start = chk_pt['epoch']
        model.load_state_dict(weight)
    
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    fake_centers_dir = os.path.join(args.save_dir, "fake_center.npy")

    if np.sum(["train_1.txt" in x for x in glob.glob(args.save_dir + "/**/*")]) == 0:
        if args.rot_only:
            create_fake_labels(None, None, args)
            
        else:
            data = dataset.Dataset(args.data, ratio=args.ratio, width=args.width, origin_width=args.origin_width, root=args.data_root, self_supervision_rot=0, mode="test", rot_bt=args.rot_bt, corruption=args.corruption, args=args)

            fake_train_loader = torch.utils.data.DataLoader(
                    data.train, batch_size=100,
                    shuffle=False, drop_last=False,
                    pin_memory=True, num_workers=args.nThreads)

            train_feature, train_labels = extract_features(model, fake_train_loader, print_freq=1e5, metric=None, pool_feature=args.pool_feature, org_feature=True)

            create_fake_labels(train_feature, train_labels, args)

            del train_feature

            fake_centers = "k-means++"

            torch.cuda.empty_cache()

    elif os.path.exists(fake_centers_dir):
        fake_centers = np.load(fake_centers_dir)
    else:
        fake_centers = "k-means++"

    time.sleep(60)
    
    model.train()
    

    

    # freeze BN
    if (args.freeze_BN is True) and (args.pretrained):
        print(40 * '#', '\n BatchNorm frozen')
        model.apply(set_bn_eval)
    else:
        print(40*'#', 'BatchNorm NOT frozen')
        
    # Fine-tune the model: the learning rate for pre-trained parameter is 1/10
    new_param_ids = set(map(id, model.module.classifier.parameters()))
    new_rot_param_ids = set()
    if args.self_supervision_rot:
        new_rot_param_ids = set(map(id, model.module.classifier_rot.parameters()))
        print(new_rot_param_ids)


    new_params = [p for p in model.module.parameters() if
                  id(p) in new_param_ids]

    new_rot_params = [p for p in model.module.parameters() if
                  id(p) in new_rot_param_ids]

    base_params = [p for p in model.module.parameters() if
                   (id(p) not in new_param_ids) and (id(p) not in new_rot_param_ids)]

    param_groups = [
                {'params': base_params},
                {'params': new_params},
                {'params': new_rot_params, 'lr': args.rot_lr}]

    print('initial model is save at %s' % save_dir)

    optimizer = torch.optim.Adam(param_groups, lr=args.lr,
                                 weight_decay=args.weight_decay)
    criterion = losses.create(args.loss, margin=args.margin, alpha=args.alpha, beta=args.beta, base=args.loss_base).cuda()
    

    data = dataset.Dataset(args.data, ratio=args.ratio, width=args.width, origin_width=args.origin_width, root=args.save_dir, self_supervision_rot=args.self_supervision_rot, rot_bt=args.rot_bt, corruption=1, args=args)
    train_loader = torch.utils.data.DataLoader(
        data.train, batch_size=args.batch_size,
        sampler=FastRandomIdentitySampler(data.train, num_instances=args.num_instances),
        drop_last=True, pin_memory=True, num_workers=args.nThreads)
    

    # save the train information

    for epoch in range(start, args.epochs):

        train(epoch=epoch, model=model, criterion=criterion,
              optimizer=optimizer, train_loader=train_loader, args=args)


        
        
        if (epoch+1) % args.save_step == 0 or epoch==0:
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_checkpoint({
                'state_dict': state_dict,
                'epoch': (epoch+1),
            }, is_best=False, fpath=osp.join(args.save_dir, 'ckp_ep' + str(epoch + 1) + '.pth.tar'))
        
        if ((epoch+1) % args.up_step == 0) and (not args.rot_only):
            # rewrite train_1.txt file
            data = dataset.Dataset(args.data, ratio=args.ratio, width=args.width, origin_width=args.origin_width, root=args.data_root, self_supervision_rot=0, mode="test", rot_bt=args.rot_bt, corruption=args.corruption, args=args)
            fake_train_loader = torch.utils.data.DataLoader(data.train, batch_size=args.batch_size, shuffle=False, drop_last=False,
            pin_memory=True, num_workers=args.nThreads)
            train_feature, train_labels = extract_features(model, fake_train_loader, print_freq=1e5, metric=None, pool_feature=args.pool_feature, org_feature=(args.dim % 64 != 0))
            fake_centers = create_fake_labels(train_feature, train_labels, args, init_centers=fake_centers)
            del train_feature
            torch.cuda.empty_cache()
            time.sleep(60)
            np.save(fake_centers_dir, fake_centers)
            # reload data
            data = dataset.Dataset(args.data, ratio=args.ratio, width=args.width, origin_width=args.origin_width, root=args.save_dir, self_supervision_rot=args.self_supervision_rot, rot_bt=args.rot_bt, corruption=1, args=args)
    

            train_loader = torch.utils.data.DataLoader(
                data.train, batch_size=args.batch_size,
                sampler=FastRandomIdentitySampler(data.train, num_instances=args.num_instances),
                drop_last=True, pin_memory=True, num_workers=args.nThreads)
            

            # test on testing data
            # extract_recalls(data=args.data, data_root=args.data_root, width=args.width, net=args.net, checkpoint=None,
            #         dim=args.dim, batch_size=args.batch_size, nThreads=args.nThreads, pool_feature=args.pool_feature,
            #         gallery_eq_query=args.gallery_eq_query, model=model)
            model.train()
            if (args.freeze_BN is True) and (args.pretrained):
                print(40 * '#', '\n BatchNorm frozen')
                model.apply(set_bn_eval)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Metric Learning')

    # hype-parameters
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate of new parameters")
    parser.add_argument('--rot_lr', type=float, default=1e-5, help="learning rate of new rot parameters")
    parser.add_argument('--batch_size', '-b', default=128, type=int, metavar='N',
                        help='mini-batch size of metric learning loss')
    parser.add_argument('--rot_batch', default=16, type=int, metavar='N',
                        help='mini-batch size of rot loss')
    parser.add_argument('--num_instances', default=8, type=int, metavar='n',
                        help=' number of samples from one class in mini-batch')
    parser.add_argument('--dim', default=512, type=int, metavar='n',
                        help='dimension of embedding space')
    parser.add_argument('--width', default=224, type=int,
                        help='width of input image')
    parser.add_argument('--origin_width', default=256, type=int,
                        help='size of origin image')
    parser.add_argument('--ratio', default=0.16, type=float,
                        help='random crop ratio for train data')
    parser.add_argument('--alpha', default=30, type=int, metavar='n',
                        help='hyper parameter in NCA and its variants')
    parser.add_argument('--beta', default=0.1, type=float, metavar='n',
                        help='hyper parameter in some deep metric loss functions')
    parser.add_argument('--orth_reg', default=0, type=float,
                        help='hyper parameter coefficient for orth-reg loss')
    parser.add_argument('-k', default=16, type=int, metavar='n',
                        help='number of neighbour points in KNN')
    parser.add_argument('--margin', default=0.5, type=float,
                        help='margin in loss function')
    parser.add_argument('--init', default='random',
                        help='the initialization way of FC layer')

    # network
    parser.add_argument('--freeze_BN', default=True, type=bool, required=False, metavar='N',
                        help='Freeze BN if True')
    parser.add_argument('--data', default='cub', required=True,
                        help='name of Data Set')
    parser.add_argument('--data_root', type=str, default=None,
                        help='path to Data Set')

    parser.add_argument('--net', default='Inception')
    parser.add_argument('--loss', default='branch', required=True,
                        help='loss for training network')
    parser.add_argument('--epochs', default=600, type=int, metavar='N',
                        help='epochs for training process')
    parser.add_argument('--save_step', default=50, type=int, metavar='N',
                        help='number of epochs to save model')

    parser.add_argument('--up_step', default=10, type=int, metavar='N',
                        help='')
    # Resume from checkpoint
    parser.add_argument('--resume', '-r', default=None,
                        help='the path of the pre-trained model')
    parser.add_argument('--pretrained', type=ast.literal_eval, default=False,
                        help='Is gallery identical with query')

    # train
    parser.add_argument('--print_freq', default=20, type=int,
                        help='display frequency of training')

    # self supervision
    parser.add_argument('--self_supervision_rot', default=0, type=float,
                        help='loss weight of supervision rotation')
    parser.add_argument('--rot_only', default=0, type=float,
                        help='only use rot')

    parser.add_argument('--rot_bt', default=1, type=int,
                        help='rotation before transform')
    parser.add_argument('--num_clusters', default=100, type=int,
                        help='num of clusters')
    parser.add_argument('--gallery_eq_query', '-g_eq_q', type=ast.literal_eval, default=False,
                        help='Is gallery identical with query')
    parser.add_argument('--pool_feature', type=ast.literal_eval, default=False, required=False,
                        help='if True extract feature from the last pool layer')

    parser.add_argument('--use_test', default=0, type=float, 
                        help='whether to use test data in self supervision')

    #label corruption, not used for now
    parser.add_argument('--corruption', default=0, type=float,
                        help='label corruption percentage')



    # basic parameter
  
    parser.add_argument('--save_dir', default=None,
                        help='where the trained models save')
    parser.add_argument('--nThreads', '-j', default=16, type=int, metavar='N',
                        help='number of data loading threads (default: 2)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)

    parser.add_argument('--loss_base', type=float, default=0.75)




    main(parser.parse_args())




