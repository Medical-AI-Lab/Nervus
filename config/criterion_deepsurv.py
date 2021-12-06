#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn



class Regularization(object):
    def __init__(self, order, weight_decay):
        ''' The initialization of Regularization class

        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        '''
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        ''' Performs calculates regularization(self.order) loss for model.

        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        '''
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss



class NegativeLogLikelihood(nn.Module):
    #def __init__(self, config, device):
    def __init__(self, device):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = 0.08   # config['l2_reg']
        # self.L2_reg = 0    # When 0, no Regularization
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)
        self.device = device

    def forward(self, risk_pred, y, e, model):
        """
        mask = torch.ones(y.shape[0], y.shape[0]).to(self.device)
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        l2_loss = self.reg(model)
        """

        mask = torch.ones(y.shape[0], y.shape[0]).to(self.device)
        mask[(y.T - y) > 0] = 0
        l1 = torch.exp(risk_pred) * mask
        l2 = torch.sum(l1, dim=0) / torch.sum(mask, dim=0)
        l3 = torch.log(l2).reshape(-1, 1)
        num_occurs = torch.sum(e)

        if num_occurs.item() == 0.0:
            # To avoid dividing with zero
            #neg_log_loss = torch.tensor([0.0], requires_grad = True)
            #neg_log_loss = torch.tensor([1.0], requires_grad = True)
            #l2_loss = self.reg(model)
            #loss = neg_log_loss + l2_loss
            loss = torch.tensor([1e-7], requires_grad = True)

        else:
            neg_log_loss = -torch.sum((risk_pred-l3) * e) / num_occurs
            l2_loss = self.reg(model)
            loss = neg_log_loss + l2_loss

        return loss
