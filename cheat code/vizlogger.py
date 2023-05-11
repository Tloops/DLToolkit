import datetime
import sys
import time

import torch as torch
from visdom import Visdom
import numpy as np
import torch


def tensor2image(tensor):
    """
    This is available only when image distribution is among [-1, 1] -> [0, 255]
    The main function of it is transferring the image from tensor to numpy...
    """
    image = (127.5 * (tensor.cpu().float().numpy())) + 127.5
    image1 = image[0]
    for i in range(1, tensor.shape[0]):
        image1 = np.hstack((image1, image[i]))
    return image1.astype(np.uint8)


class Logger:
    """
    a wrapper class for visdom providing basic display functions for training
    """

    def __init__(self, env_name, port, n_epochs, batches_epoch):
        """
        :param env_name: basic setting for visdom
        :param port: basic setting for visdom
        :param n_epochs: maximum of epoch
        :param batches_epoch: how many batches are there in the dataloader, i.e. len(dataloader)
        """
        self.viz = Visdom(port=port, env=env_name)
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, losses=None, images=None):
        """
        :param losses: dictionary for name and value of each loss
        :param images: dictionary for image to be visualized
        """
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i + 1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch))
        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title': image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name],
                               opts={'title': image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]),
                                                                 Y=np.array([loss / self.batch]),
                                                                 opts={'xlabel': 'epochs', 'ylabel': loss_name,
                                                                       'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss / self.batch]),
                                  win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


if __name__ == "__main__":
    max_epoch = 100
    logger = Logger(env_name="main", port=6677, n_epochs=max_epoch, batches_epoch=10)
    for i in range(max_epoch):
        for j in range(10):
            img = torch.rand(1, 3, 256, 256)  # random image
            loss_x, loss_y = torch.rand(1), torch.rand(1)  # random loss
            logger.log(losses={"lossX": loss_x, "lossY": loss_y}, images={"train_img": img})
