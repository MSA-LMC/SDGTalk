#!/usr/bin/env python

import os
import sys
import torch
import pickle
import numpy as np

from engines import LMDBEngine, CoreEngine

class Tracker:
    def __init__(self, focal_length, device='cuda'):
        self._device = device
        self.tracker = CoreEngine(focal_length=focal_length, device=device)

    def track_lmdb(self, lmdb_path, dir_path=None):
        # build name
        data_name = os.path.basename(lmdb_path[:-1] if lmdb_path.endswith('/') else lmdb_path)
        output_path = os.path.join('outputs_mead', dir_path) if dir_path else f'outputs_mead/{data_name}'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print('Load lmdb data...')
        lmdb_engine = LMDBEngine(lmdb_path, write=False)
        print('Track with flame/bbox...')
        base_results = self.tracker.track_base(lmdb_engine, output_path)
        print('Track with flame/bbox done!')
        print('Track optim...')
        optim_results = self.tracker.track_optim(base_results, output_path, lmdb_engine, share_id=False)
        print('Track optim done!')
        lmdb_engine.close()


if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=UserWarning, lineno=0, append=False)
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path', '-l', default='img_lmdb/', type=str)
    parser.add_argument('--outdir_path', '-d', default='', type=str)
    parser.add_argument('--split_id', '-s', default=0, type=int)
    args = parser.parse_args()
    
    tracker = Tracker(focal_length=12.0, device='cuda')
    tracker.track_lmdb(args.lmdb_path, dir_path=args.outdir_path)
   
