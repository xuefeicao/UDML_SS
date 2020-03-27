from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class WeightLoss(nn.Module):
    def __init__(self, alpha=50, beta=2, margin=0.5, hard_mining=True, **kwargs):
        super(WeightLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining

    def forward(self, inputs, targets):
        n = inputs.size(0)
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets

        base = 0.5
        loss = []
        c = 0

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets==targets[i])
            org_pos_pair = pos_pair_

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            org_pos_pair_1 = pos_pair_
            neg_pair_ = torch.masked_select(sim_mat[i], targets!=targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]
            if len(pos_pair_) == 0:
                c += 1
                continue

            if self.hard_mining is not None:
                try:
                    neg_pair = torch.masked_select(neg_pair_, neg_pair_ + 0.1 > pos_pair_[0])
                    pos_pair = torch.masked_select(pos_pair_, pos_pair_ - 0.1 <  neg_pair_[-1])
                    
                    if len(neg_pair) < 1 or len(pos_pair) < 1:
                        c += 1
                        continue

                    pos_loss = 1.0/self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (pos_pair - base))))
                    neg_loss = 1.0/self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha * (neg_pair - base))))
                    loss.append(pos_loss + neg_loss)
                except Exception as e:
                    raise Exception((neg_pair_.size(), pos_pair_.size(), org_pos_pair, org_pos_pair_1, sim_mat[i,0:5], inputs[0:5, 0:5]))


            else:
                # print('hello world')
                neg_pair = neg_pair_
                pos_pair = pos_pair_
                pos_loss = 1.0/self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (pos_pair - base))))
                neg_loss = 1.0/self.alpha * torch.log(1 + torch.sum(torch.exp(self.alpha * (neg_pair - base))))

                loss.append(pos_loss + neg_loss)
            
        loss = sum(loss)/n
        prec = float(c)/n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        return  loss, prec, mean_pos_sim, mean_neg_sim

def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    # margin = 0.5
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    # print(x)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(WeightLoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')