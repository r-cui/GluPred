import argparse
import os
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        self.parser.add_argument("--seed", type=int, default=0,
                                 help="Random seed for torch.")

        # general options
        self.parser.add_argument("--data_dir", type=str, default="./data")
        self.parser.add_argument("--left_len", type=int, default=24)
        self.parser.add_argument("--missing_len", type=int, default=6)
        self.parser.add_argument("--patient", type=str, default="559")
        self.parser.add_argument("--ckpt", type=str, default="./checkpoint/",
                                 help="The directory (folder) to save trained models.")

        # experiment setup
        self.parser.add_argument("--single_pred", action='store_true',
                                 help="If true, predict CGM and assume others known; if false, predict all features.")
        self.parser.add_argument("--unimodal", action='store_true',
                                 help="If true, use glucose only; if false, use multiple features.")
        self.parser.add_argument("--transfer_learning", action='store_true',
                                 help="If true, use transfer learning; if false, train on personal data only.")

        # model options
        self.parser.add_argument("--num_layers", type=int, default=3)
        self.parser.add_argument("--d_model", type=int, default=128)
        self.parser.add_argument("--heads", type=int, default=4)
        self.parser.add_argument("--d_ff", type=int, default=512)
        self.parser.add_argument("--dropout", type=float, default=0.1)
        self.parser.add_argument("--attention_dropout", type=float, default=0.1)

        # training option
        self.parser.add_argument('--pre_lr', type=float, default=0.0003)
        self.parser.add_argument('--pre_epoch', type=int, default=10)
        self.parser.add_argument('--ft_lr', type=float, default=0.00005)
        self.parser.add_argument('--epoch', type=int, default=10)
        self.parser.add_argument('--decay_factor', type=float, default=0.5)
        self.parser.add_argument('--decay_patience', type=int, default=1)
        self.parser.add_argument('--batch_size', type=int, default=16)

    def print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()

        if not os.path.exists(self.opt.ckpt):
            os.makedirs(self.opt.ckpt)

        self.print()
        return self.opt
