"""
Minimizing Mahalanobis distance between related pairs, and maximizing between negative pairs.

A loss typically used for creating a Euclidian embedding space for a wide variety of supervised learning problems.
The original implementation was by Davis King @ Dlib.

PyTorch Implementation: https://gist.github.com/bkj/565c5e145786cfd362cffdbd8c089cf4

Made changes such that accuracy is provided on a forward pass as well.
"""

import torch
import torch.nn.functional as F
from torch import nn
import time


class MahalanobisMetricLoss(nn.Module):
    def __init__(self, margin=0.6, extra_margin=0.04):
        super(MahalanobisMetricLoss, self).__init__()

        self.margin = margin
        self.extra_margin = extra_margin

    def forward(self, outputs, targets):
        """
        :param outputs: Outputs from a network. (sentence_batch size, # features)
        :param targets: Target labels. (sentence_batch size, 1)
        :param margin: Minimum distance margin between contrasting sample pairs.
        :param extra_margin: Extra acceptable margin.
        :return: Loss and accuracy. Loss is a variable which may have a backward pass performed.
        """

        loss = torch.zeros(1)
        if torch.cuda.is_available(): loss = loss.cuda()
        loss = torch.autograd.Variable(loss)

        batch_size = outputs.size(0)

        # Compute Mahalanobis distance matrix.
        magnitude = (outputs ** 2).sum(1).expand(batch_size, batch_size)
        squared_matrix = outputs.mm(torch.t(outputs))
        mahalanobis_distances = F.relu(magnitude + torch.t(magnitude) - 2 * squared_matrix).sqrt()

        # Determine number of positive + negative thresholds.
        neg_mask = targets.expand(batch_size, batch_size)
        neg_mask = (neg_mask - neg_mask.transpose(0, 1)) != 0

        num_pairs = (1 - neg_mask).sum()  # Number of pairs.
        num_pairs = (num_pairs - batch_size) / 2  # Number of pairs apart from diagonals.
        num_pairs = num_pairs.data[0]

        negative_threshold = mahalanobis_distances[neg_mask].sort()[0][num_pairs].data[0]

        num_right, num_wrong = 0, 0

        for row in range(batch_size):
            for column in range(batch_size):
                x_label = targets[row].data[0]
                y_label = targets[column].data[0]
                mahalanobis_distance = mahalanobis_distances[row, column]
                euclidian_distance = torch.dist(outputs[row], outputs[column])

                if x_label == y_label:
                    # Positive examples should be less than (margin - extra_margin).
                    if mahalanobis_distance.data[0] > self.margin - self.extra_margin:
                        loss += mahalanobis_distance - (self.margin - self.extra_margin)

                    # Compute accuracy w/ Euclidian distance.
                    if euclidian_distance.data[0] < self.margin:
                        num_right += 1
                    else:
                        num_wrong += 1
                else:
                    # Negative examples should be greater than (margin + extra_margin).
                    if (mahalanobis_distance.data[0] < self.margin + self.extra_margin) and (
                                mahalanobis_distance.data[0] < negative_threshold):
                        loss += (self.margin + self.extra_margin) - mahalanobis_distance

                    # Compute accuracy w/ Euclidian distance.
                    if euclidian_distance.data[0] < self.margin:
                        num_wrong += 1
                    else:
                        num_right += 1

        accuracy = num_right / (num_wrong + num_right)
        return loss / (2 * num_pairs), accuracy


def tensor_correlation(output, target):
    x = output
    y = target

    xy = x * y
    mean_xy = torch.mean(xy)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    cov_xy = mean_xy - mean_x * mean_y

    var_x = torch.sum((x - mean_x) ** 2 / x.shape[0])
    var_y = torch.sum((y - mean_y) ** 2 / y.shape[0])

    corr_xy = cov_xy / (torch.sqrt(var_x * var_y))
    return corr_xy


def get_correlation_loss(labels, outputs, batch_size, dof_based=False, case_based=False):
    """
    :param labels: a tensor of labels N * 6
    :param outputs: a tensor of outputs N * 6
    :param batch_size: how many cases are there in each batch (no regard to duplicate samples)
    :param dof_based: correlation is calculated through each degree of freedom
    :param case_based: correlation is calculated through each case
    :return: tensor of correlation loss
    """
    # print('labels shape {}, outputs shape {}'.format(labels.shape, outputs.shape))

    if dof_based:
        if case_based:
            case_samples_num = int(labels.shape[0] / batch_size)
            # print(case_samples_num)
            batch_corr = []
            for batch_case_id in range(batch_size):
                start_index = batch_case_id * case_samples_num
                end_index = (batch_case_id + 1) * case_samples_num
                case_corr = []
                for dof_id in range(labels.shape[1]):
                    corr = tensor_correlation(output=outputs[start_index:end_index, dof_id],
                                              target=labels[start_index:end_index, dof_id])
                    case_corr.append(corr)
                    # time.sleep(30)
                case_corr = sum(case_corr) / labels.shape[1]
                batch_corr.append(case_corr)
            # print('len batch_corr {}'.format(len(batch_corr)))
            batch_corr = sum(batch_corr) / batch_size
            loss = 1 - batch_corr
            # print('loss {}'.format(loss))
            # time.sleep(30)
        else:
            dof_correlation = []
            for dof_id in range(labels.shape[1]):
                x = outputs[:, dof_id]
                y = labels[:, dof_id]

                xy = x * y
                mean_xy = torch.mean(xy)
                mean_x = torch.mean(x)
                mean_y = torch.mean(y)
                cov_xy = mean_xy - mean_x * mean_y

                var_x = torch.sum((x - mean_x) ** 2 / x.shape[0])
                var_y = torch.sum((y - mean_y) ** 2 / y.shape[0])

                corr_xy = cov_xy / (torch.sqrt(var_x * var_y))

                loss = 1 - corr_xy
                dof_correlation.append(loss)
            loss = sum(dof_correlation) / 6

    else:
        x = outputs.flatten()
        y = labels.flatten()
        # print('x shape {}, y shape {}'.format(x.shape, y.shape))
        # print('x shape\n{}\ny shape\n{}'.format(x, y))
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
    # print('return loss {}'.format(loss))
    # time.sleep(30)
    return loss


def dof_MSE(labels, outputs, criterion, dof_based=False):
    if dof_based:
        dof_losses = []
        for dof_id in range(labels.shape[1]):
            # print(labels[:, dof_id].shape)
            x = outputs[:, dof_id]
            y = labels[:, dof_id]

            dof_loss = criterion(x, y)
            dof_losses.append(dof_loss)
        print(dof_losses)
        loss = sum(dof_losses) / 6
        print(loss)
        print(criterion(labels, outputs))
        time.sleep(30)
    else:
        loss = criterion(labels, outputs)

    return loss





























