#todo add
import torch
import torch.nn as nn
import torch.nn.functional as F

from IP_osr_patches import parser
from loss.Dist import Dist
import argparse


parser.add_argument('--lbda', default = 0.1, type = float, help='Weighting of Anchor loss component')
args = parser.parse_args()


class CACLoss():
    def __init__(self, **options):
        self.num_known_classes = options['known']
        self.use_gpu = options['use_gpu']

    def forward(self, distances, gt):  # todo 特征,预测类别，真实标签
        '''Returns CAC loss, as well as the Anchor and Tuplet loss components separately for visualisation.'''
        true = torch.gather(distances, 1, gt.view(-1, 1)).view(-1)
        non_gt = torch.Tensor(
            [[i for i in range(self.num_known_classes) if gt[x] != i] for x in range(len(distances))]).long().cuda()
        others = torch.gather(distances, 1, non_gt)

        anchor = torch.mean(true)

        tuplet = torch.exp(-others + true.unsqueeze(1))
        tuplet = torch.mean(torch.log(1 + torch.sum(tuplet, dim=1)))

        total = args.lbda * anchor + tuplet


        return total, anchor, tuplet

# def CACLoss(distances, gt):
#     '''Returns CAC loss, as well as the Anchor and Tuplet loss components separately for visualisation.'''
#     true = torch.gather(distances, 1, gt.view(-1, 1)).view(-1)
#     non_gt = torch.Tensor(
#         [[i for i in range(cfg['num_known_classes']) if gt[x] != i] for x in range(len(distances))]).long().cuda()
#     others = torch.gather(distances, 1, non_gt)
#
#     anchor = torch.mean(true)
#
#     tuplet = torch.exp(-others + true.unsqueeze(1))
#     tuplet = torch.mean(torch.log(1 + torch.sum(tuplet, dim=1)))
#
#     total = args.lbda * anchor + tuplet
#
#
#     return total, anchor, tuplet
