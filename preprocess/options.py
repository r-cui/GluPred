import argparse
import os


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        self.parser.add_argument("--data_folder_path", type=str, default='../../OHIO_Data/',
                                 help='Please add the path to the folder containing the OhioT1DM dataset')
        self.parser.add_argument("--extract_folder_path", type=str, default='temp/',
                                 help='Please add a valid folder to extract data.')

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        if not os.path.exists(self.opt.extract_folder_path):
            os.makedirs(self.opt.extract_folder_path)
        return self.opt