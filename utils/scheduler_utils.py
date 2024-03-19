#!/usr/bin/env python
# coding=utf-8
import os
import torch


class StepLRSchedule():
    def __init__(self, lr_group_init, gamma, step_size):
        self.lr_group_init = lr_group_init
        self.gamma = gamma
        self.step_size = step_size

    def get_lr_group(self, epoch):
        e = epoch // self.step_size
        lr_group = []
        for lr in self.lr_group_init:
            lr_group.append(lr * (self.gamma ** e))
        return lr_group


class MultiplicativeLRSchedule():
    def __init__(self, lr_group_init, gammas, milestones):
        '''
        Args:
            milestones: list, epoch in increasing order
            gammas: list 
        '''
        assert(len(gammas)==len(milestones))
        self.lr_group_init = lr_group_init
        self.gammas = gammas
        self.milestones = milestones

    def get_lr_group(self, epoch):
        factor = 1.
        for g, m in zip(self.gammas, self.milestones):
            if epoch >= m:
                factor *= g
            else:
                break

        lr_group = []
        for lr in self.lr_group_init:
            lr_group.append(lr * factor)
        return lr_group


def adjust_learning_rate(lr_scheduler, optimizer, epoch):
    lr_group = lr_scheduler.get_lr_group(epoch)
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_group[i]
    return lr_group




