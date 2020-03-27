# coding=utf-8
from __future__ import print_function, absolute_import
import time
from utils import AverageMeter, orth_reg
import  torch
import numpy as np
from torch.autograd import Variable
from torch.backends import cudnn
from torch import nn

cudnn.benchmark = True


def train(epoch, model, criterion, optimizer, train_loader, args):
    losses = AverageMeter()
    batch_time = AverageMeter()
    accuracy = AverageMeter()
    pos_sims = AverageMeter()
    neg_sims = AverageMeter()

    end = time.time()

    freq = min(args.print_freq, len(train_loader))
    if not args.use_test or True:
        test_loader = [(0, 0) for _ in range(len(train_loader))]
    else:
        test_loader = args.use_test
    for i, (data_, test_data_) in enumerate(zip(train_loader, test_loader), 0):
        inputs, labels = data_
        inputs_test, _ = test_data_
        num_samples, _, w, h = inputs.size()
        
        inputs_1 = inputs[:, 0:3, :, :]
        inputs_2 = inputs[np.random.choice(range(num_samples), args.rot_batch), :, :, :].view(-1, 3, w, h)
        if args.use_test and False:
            num_samples = inputs_test.size(0)
            inputs_3 = inputs_test[np.random.choice(range(num_samples), args.rot_batch), :, :, :].view(-1, 3, w, h)
            inputs_3 = Variable(inputs_3).cuda()
        # wrap them in Variable

        inputs_1 = Variable(inputs_1).cuda()
        inputs_2 = Variable(inputs_2).cuda()
        labels = Variable(labels).cuda()
        
        optimizer.zero_grad()


        

        if not args.rot_only:
            embed_feat = model(inputs_1, rot=False)
            if args.dim % 64 != 0:
                loss, inter_, dist_ap, dist_an = nn.CrossEntropyLoss()(embed_feat, labels), 0, 0, 0
            else:
                loss, inter_, dist_ap, dist_an = criterion(embed_feat, labels)
        else:
            loss, inter_, dist_ap, dist_an = 0, 0, 0, 0

        loss_rot = torch.zeros(1)
        loss_rot_test = torch.zeros(1)
        if args.self_supervision_rot:
            score = model(inputs_2, rot=True)
            labels_rot = torch.LongTensor([0, 1, 2, 3] * args.rot_batch).cuda()
            loss_rot = nn.CrossEntropyLoss()(score, labels_rot)
            loss += args.self_supervision_rot * loss_rot


            


        if args.orth_reg != 0:
            loss = orth_reg(net=model, loss=loss, cof=args.orth_reg)

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if not args.rot_only:
            losses.update(loss.item())
            accuracy.update(inter_)
            pos_sims.update(dist_ap)
            neg_sims.update(dist_an)

        
        if (i + 1) % freq == 0 or (i+1) == len(train_loader):
            print('Epoch: [{0:03d}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f} \t'
                  'Loss_rot {loss_rot:.4f} \t'
                  'Loss_rot_test {loss_rot_test:.4f} \t'
                  'accuracy {accuracy.avg:.4f} \t'
                  'Pos {pos.avg:.4f}\t'
                  'Neg {neg.avg:.4f} \t'.format
                  (epoch + 1, i + 1, len(train_loader), batch_time=batch_time,
                   loss=losses, loss_rot=loss_rot.item(), loss_rot_test=loss_rot_test.item(), accuracy=accuracy, pos=pos_sims, neg=neg_sims))

