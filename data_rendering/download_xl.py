import os
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict

if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=10)
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--meta_name', type=str, default='metadata.csv')
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(opt.output_dir, exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, opt.meta_name)):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, opt.meta_name))
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'local_path' in metadata.columns:
            metadata = metadata[metadata['local_path'].isna()]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    total_range = opt.end_idx - opt.start_idx
    start = opt.start_idx + (total_range * opt.rank // opt.world_size)
    end = opt.start_idx + (total_range * (opt.rank + 1) // opt.world_size)
    metadata = metadata[start:end]
                
    print(f'Processing {len(metadata)} objects...')

    # process objects
    downloaded = dataset_utils.download(metadata, **opt)
    downloaded.to_csv(os.path.join(opt.output_dir, f'downloaded_{opt.rank}.csv'), index=False)
