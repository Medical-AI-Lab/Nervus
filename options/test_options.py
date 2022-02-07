#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import argparse



class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Test options')

        self.parser.add_argument('--test_batch_size', type=int, default=64,   metavar='N', help='batch size for test (Default: 64)')
        self.parser.add_argument('--test_datetime',   type=str, default=None, help='datetime when training (Default: None)')
        self.parser.add_argument('--normalize_image', type=str, default='yes', help='image nomalization, yes no no (Default: yes)')

    def parse(self):
        self.args = self.parser.parse_args()
        return vars(self.args)

# ----- EOF -----