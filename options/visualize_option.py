#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import argparse



class VisualizeOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Visualize options')

        self.parser.add_argument('--visualization_datetime', type=str, default=None, help='datetime for visualization (Default: None)')
        self.parser.add_argument('--visualization_split',  type=str, default='train,val,test', help='split to be visualized: eg. train,val,test, val,test. (Default: train,val,test)')

    def parse(self):
        self.args = self.parser.parse_args()
        return vars(self.args)

# ----- EOF -----