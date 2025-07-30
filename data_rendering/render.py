import os
import subprocess
import multiprocessing
import logging
import random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--obj_path", type=str, default="./obj_data/hf-objaverse-v1/glbs")
parser.add_argument("--save_dir", type=str, default='./output')
parser.add_argument("--gpu_num", type=int, default=8)
parser.add_argument("--frame_num", type=int, default=24)
parser.add_argument("--view_num", type=int, default=100)
parser.add_argument("--azimuth_aug", type=int, default=0)
parser.add_argument("--elevation_aug", type=int, default=0)
parser.add_argument("--resolution", default=512)
parser.add_argument("--mode_multi", type=int, default=0)
parser.add_argument("--mode_static", type=int, default=0)
parser.add_argument("--mode_front_view", type=int, default=1)
parser.add_argument("--mode_four_view", type=int, default=0)
parser.add_argument("--txt_file", type=str, default='/mnt/blob/data/obj_xl_4d_clean.txt')
parser.add_argument("--start_idx", type=int, default=0)
parser.add_argument("--end_idx", type=int, default=6400)
parser.add_argument("--workers_per_gpu", type=int, default=8)
parser.add_argument("--num_augment", type=int, default=1)
parser.add_argument("--overwrite", action='store_true')
args = parser.parse_args()

def worker(queue, count, gpu_id):
    while True:
        item = queue.get()
        if item is None:
            break
        
        file_name = os.path.basename(item).split('.')[0]
        save_path = os.path.join(args.save_dir, file_name)
        for i in range(args.num_augment):
            output_dir = os.path.join(save_path, f"{i-1}") if i > 0 else save_path
            command = f'CUDA_VISIBLE_DEVICES={gpu_id} && blender-3.2.2-linux-x64/blender \
                --background --python blender_new.py -- \
                --object_path {item} \
                --frame_num {args.frame_num} \
                --view_num {args.view_num} \
                --output_dir {output_dir} \
                --gpu_id {gpu_id} \
                --resolution {args.resolution} \
                --mode_multi {args.mode_multi} \
                --mode_static {args.mode_static} \
                --mode_front {args.mode_front_view} \
                --mode_four_view {args.mode_four_view} \
                '
            if i > 0:
                command += ' --augment'
            if args.overwrite:
                command += ' --overwrite'

            print('command:', command)
            logging.info(f'Executing command: {command}')
            try:
                result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logging.info(result.stdout.decode())
                logging.error(result.stderr.decode())
            except Exception as e:
                print(f"An error occurred while running command: {command}")
                print(str(e))

        with count.get_lock():
            count.value += 1
            
        queue.task_done()

# Read input files
with open(args.txt_file, 'r') as f:
    glb_prefixes = f.read().splitlines()[args.start_idx:args.end_idx]
glb_prefixes = [os.path.join(args.obj_path, file) for file in glb_prefixes]
extensions = ['DAE', 'Fbx', 'fbx', 'glb', 'FBX', 'gltf', 'dae']
glb_files = []
for glb_prefix in glb_prefixes:
    for extension in extensions:
        if os.path.exists(f"{glb_prefix}.{extension}"):
            glb_files.append(f"{glb_prefix}.{extension}")
            break
# import glob
# glb_files = glob.glob(os.path.join(args.obj_path, '**/*.glb'))
print('found glb files:', len(glb_files))
os.makedirs(args.save_dir, exist_ok=True)

if __name__ == "__main__":
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value('i', 0)
    
    # Start worker processes on each GPU
    for gpu_i in range(args.gpu_num):
        for worker_i in range(args.workers_per_gpu):
            process = multiprocessing.Process(
                target=worker,
                args=(queue, count, gpu_i)
            )
            process.daemon = True
            process.start()

    # Add items to queue
    for item in glb_files:
        queue.put(item)

    # Wait for completion
    queue.join()

    # Add sentinels to stop workers
    for i in range(args.gpu_num * args.workers_per_gpu):
        queue.put(None)
