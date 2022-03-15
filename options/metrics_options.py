#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

class MetricsOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Metrics options')

        self.parser.add_argument('--likelihood_datetime', type=str, default=None, help='datetime when training (Default: None)')

    def parse(self):
        self.args = self.parser.parse_args()

        return vars(self.args)

# ----- EOF -----
