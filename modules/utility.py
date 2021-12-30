import os

import numpy as np
import torch
import cv2


class checkpoint:
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        self.dir = args.save_name
        os.makedirs(self.dir, exist_ok=True)

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save_results(self, filename, save_list):
        filename = self.get_path('{}'.format(filename))
        for v in save_list:
            img = np.uint8(v[0].detach().cpu().numpy().transpose(1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite('{}.png'.format(filename), img)


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
