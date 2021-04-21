import os
import torch


def save_ckpt(obj, opt):
    torch.save(obj, os.path.join(opt.ckpt, "{}.ckpt".format(opt.patient)))

