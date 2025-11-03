import os
import torch
import numpy as np
import trimesh
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_mesh_vertices(mesh_root, node_num=8192, timesteps=24):
    # Load animated mesh from glb
    static_mesh = trimesh.load(f"{mesh_root}/frame0.obj")
    
    # Initialize moving_frame_vertices list
    moving_frame_vertices = []
    
    # Sample points on static mesh surface
    sampled_points, face_indices = trimesh.sample.sample_surface(static_mesh, node_num)
    sampled_points = torch.tensor(sampled_points, dtype=torch.float32)
    static_frame_vertices = sampled_points
    
    # Get barycentric coordinates for interpolation
    triangles = static_mesh.triangles[face_indices]
    bary_coords = trimesh.triangles.points_to_barycentric(triangles, sampled_points)
    
    # Convert to tensors
    triangles = torch.tensor(triangles, dtype=torch.float32) 
    bary_coords = torch.tensor(bary_coords, dtype=torch.float32)
    face_indices = torch.tensor(face_indices, dtype=torch.long)
    
    # For each frame in the animation
    for i in range(1, timesteps):
        dynamic_mesh = trimesh.load(f"{mesh_root}/frame{i}.obj")
        frame_triangles = dynamic_mesh.vertices[dynamic_mesh.faces[face_indices]]
        interpolated_points = (
            bary_coords[..., None] * frame_triangles
        ).sum(dim=1)
        moving_frame_vertices.append(torch.tensor(interpolated_points, dtype=torch.float32))

    moving_frame_vertices.insert(0, sampled_points)
    moving_frame_vertices = torch.stack(moving_frame_vertices, dim=0)

    canonical_frame = 0
    canonical_frame_vertices = moving_frame_vertices[canonical_frame]

    moving_frame_deltas = []
    for timestep in range(timesteps):
        moving_frame_delta = moving_frame_vertices[timestep] - canonical_frame_vertices
        moving_frame_deltas.append(moving_frame_delta)
    
    moving_frame_deltas = torch.stack(moving_frame_deltas, dim=0)

    return canonical_frame_vertices, moving_frame_deltas, moving_frame_vertices, canonical_frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_root", type=str, default="/mnt/blob/data/objaverse_4d_rendering_no_light/")
    parser.add_argument("--csv_file", type=str, default="data/metadata.csv")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=24)
    args = parser.parse_args()

    metadata = pd.read_csv(args.csv_file)
    lines = metadata["sha256"].tolist()[args.start_idx:args.end_idx]

    def process_single_mesh(line, args):
        mesh_root = os.path.join(args.mesh_root, line.strip(), "objs")
        if not os.path.exists(os.path.join(args.mesh_root, line.strip())):
            print("Skipping ", line.strip(), " - mesh not found")
            return (line.strip(), None)
        if os.path.exists(f"{mesh_root}/mesh_data_{args.timesteps}/static_frame_vertices.pt") and os.path.exists(f"{mesh_root}/mesh_data_{args.timesteps}/moving_frame_deltas.pt") and os.path.exists(f"{mesh_root}/mesh_data_{args.timesteps}/moving_frame_vertices.pt"):
            print("Skipping ", line.strip(), " - vertices already exist")
            return (line.strip(), None)
        static_frame_vertices, moving_frame_deltas, moving_frame_vertices, canonical_frame = load_mesh_vertices(mesh_root, timesteps=args.timesteps)
        os.makedirs(f"{mesh_root}/mesh_data_{args.timesteps}/", exist_ok=True)
        torch.save(static_frame_vertices, f"{mesh_root}/mesh_data_{args.timesteps}/static_frame_vertices.pt")
        torch.save(moving_frame_deltas, f"{mesh_root}/mesh_data_{args.timesteps}/moving_frame_deltas.pt")
        torch.save(moving_frame_vertices, f"{mesh_root}/mesh_data_{args.timesteps}/moving_frame_vertices.pt")
        print("Finish loading mesh: ", line.strip(), "with shape: ", static_frame_vertices.shape, moving_frame_deltas.shape, moving_frame_vertices.shape)
        return (line.strip(), canonical_frame)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_mesh, line, args) for line in lines]
        for future in as_completed(futures):
            try:
                mesh_id, canonical_frame = future.result()
                if canonical_frame is not None:
                    canonical_frames_dict[mesh_id] = canonical_frame
            except Exception as e:
                print(f"Error processing mesh: {e}")
