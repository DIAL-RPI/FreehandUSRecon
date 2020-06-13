# -*- coding: utf-8 -*-
"""
Loss functions designed by myself, for better training the networks
The input should be 2 batch_size x n_dimensional vector: network outputs and labels
"""

# %%

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import time


# %%
''' Correlation loss for evaluator '''
def correlation_loss(output, target):
    x = output.flatten()
    y = target
    # print('x shape {}, y shape {}'.format(x.shape, y.shape))
    xy = x * y
    mean_xy = torch.mean(xy)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    cov_xy = mean_xy - mean_x * mean_y
    # print('xy shape {}'.format(xy.shape))
    # print('xy {}'.format(xy))
    # print('mean_xy {}'.format(mean_xy))
    # print('cov_xy {}'.format(cov_xy))

    var_x = torch.sum((x - mean_x) ** 2 / x.shape[0])
    var_y = torch.sum((y - mean_y) ** 2 / y.shape[0])
    # print('var_x {}'.format(var_x))

    corr_xy = cov_xy / (torch.sqrt(var_x * var_y))
    # print('correlation_xy {}'.format(corr_xy))

    loss = 1 - corr_xy
    # time.sleep(30)
    # x = output
    # y = target
    #
    # vx = x - torch.mean(x)
    # vy = y - torch.mean(y)
    #
    # loss = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    # print(loss)
    return loss

def correlation_loss_np(output, target):
    # output = output.data.cpu().numpy()
    # target = target.data.cpu().numpy()
    # output = output.flatten()
    print('output {}, target {}'.format(output.shape, target.shape))
    correlation = np.corrcoef(output, target)[0, 1]
    # loss = 1 - correlation

    return correlation

if __name__ == '__main__':
    x = np.linspace(1, 50, num=50)
    y = 3 * x
    print(x)
    print(y)
    # loss = correlation_loss(output=x, target=y)
    # print('loss = {:.4f}'.format(loss))
