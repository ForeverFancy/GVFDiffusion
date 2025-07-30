import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
from utils import sphere_hammersley_sequence


BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


def _render(file_path, sha256, output_dir, num_views, frame_num, gpu_id, overwrite=False, uniform_sampling=False, num_augment=5):
    output_folder = os.path.join(output_dir, 'cond_renders', sha256)
    
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    for i in range(num_augment):
        output_dir = os.path.join(output_folder, f'{i-1}') if i > 0 else output_folder
        args = [
            BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_new.py'),
            '--',
            # '--views', json.dumps(views),
            # '--object_path', file_path,
            '--object_path', os.path.expanduser(file_path),
            '--frame_num', str(frame_num),
            '--view_num', str(num_views),
            '--output_dir', output_dir,
            '--gpu_id', str(gpu_id),
            '--resolution', '512',
            '--mode_multi', '0',
            '--mode_static', '0',
            '--mode_front', '1',
            '--mode_four_view', '0',
        ]
        if i > 0:
            args.append('--augment')
        if overwrite:
            args.append('--overwrite')
        if uniform_sampling:
            args.append('--uniform_sampling')
        print(args)
        if file_path.endswith('.blend'):
            args.insert(1, file_path)
        
        # Set CUDA_VISIBLE_DEVICES environment variable before calling blender
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        call(args, stdout=DEVNULL, stderr=DEVNULL, env=env)

    
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return {'sha256': sha256, 'rendered': True}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'ObjaverseXL')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--frame_num', type=int, default=24,
                        help='Number of frames to render')
    parser.add_argument('--num_views', type=int, default=1,
                        help='Number of views to render')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=10)
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--uniform_sampling', action='store_true', help='Uniform sampling of frames')
    parser.add_argument('--meta_name', type=str, default='metadata.csv', help='Name of the metadata file')
    parser.add_argument('--num_augment', type=int, default=5, help='Number of augmentations to render')
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, 'renders'), exist_ok=True)
    
    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, opt.meta_name)):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, opt.meta_name))
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'rendered' in metadata.columns:
            metadata = metadata[metadata['rendered'] == False]
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
    records = []

    # filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json')):
            records.append({'sha256': sha256, 'rendered': True})
            metadata = metadata[metadata['sha256'] != sha256]
                
    print(f'Processing {len(metadata)} objects...')

    # Reset index to use it for GPU assignment
    metadata = metadata.reset_index(drop=True)
    
    # Process objects with GPU ID based on file index
    def get_gpu_id(idx):
        return idx % opt.num_gpus
    
    # Group files by GPU for better logging
    for gpu_id in range(opt.num_gpus):
        gpu_files = metadata[metadata.index % opt.num_gpus == gpu_id]
        if len(gpu_files) > 0:
            print(f'GPU {gpu_id} will process {len(gpu_files)} files')
    
    # Process objects
    func = partial(_render, output_dir=opt.output_dir, num_views=opt.num_views, frame_num=opt.frame_num, overwrite=opt.overwrite, uniform_sampling=opt.uniform_sampling, num_augment=opt.num_augment)
    rendered = dataset_utils.foreach_instance(
        metadata, 
        opt.output_dir, 
        lambda file_path, sha256: func(file_path, sha256, gpu_id=get_gpu_id(metadata[metadata['sha256'] == sha256].index[0])),
        max_workers=opt.max_workers, 
        desc=f'Rendering objects across {opt.num_gpus} GPUs'
    )
    rendered = pd.concat([rendered, pd.DataFrame.from_records(records)])
    rendered.to_csv(os.path.join(opt.output_dir, f'rendered_{opt.rank}.csv'), index=False)
