import argparse
import os
import time
import numpy as np
from tqdm import tqdm
import open3d as o3d
from openfusion.slam import build_slam, BaseSLAM
from openfusion.datasets import Dataset
from openfusion.utils import (
    show_pc, save_pc, get_cmap_legend
)
from configs.build import get_config


def stream_loop(args, slam:BaseSLAM):
    if args.save:
        slam.export_path = f"{args.data}_live/{args.algo}.npz"

    slam.start_thread()
    if args.live:
        slam.start_monitor_thread()
        slam.start_query_thread()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        slam.stop_thread()
        if args.live:
            slam.stop_query_thread()
            slam.stop_monitor_thread()


def dataset_loop(args, slam:BaseSLAM, dataset:Dataset):
    if args.save:
        slam.export_path = f"{args.data}_{args.scene}_{args.algo}.npz"

    if args.live:
        slam.start_monitor_thread()
        slam.start_query_thread()
    i = 0
    for rgb_path, depth_path, extrinsics in tqdm(dataset):
        rgb, depth = slam.io.from_file(rgb_path, depth_path)
        slam.io.update(rgb, depth, extrinsics)
        slam.vo()
        slam.compute_state(encode_image=i%10==0)
        i += 1
    if args.live:
        slam.stop_query_thread()
        slam.stop_monitor_thread()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default="vlfusion", choices=["default", "cfusion", "vlfusion"])
    parser.add_argument('--vl', type=str, default="seem", help="vlfm to use")
    parser.add_argument('--data', type=str, default="kobuki", help='Path to dir of dataset.')
    parser.add_argument('--scene', type=str, default="icra", help='Name of the scene in the dataset.')
    parser.add_argument('--frames', type=int, default=-1, help='Total number of frames to use. If -1, use all frames.')
    parser.add_argument('--device', type=str, default="cuda:0", choices=["cpu:0", "cuda:0"])
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--host_ip', type=str, default="YOUR IP") # for stream
    args = parser.parse_args()

    params = get_config(args.data, args.scene)
    dataset:Dataset = params["dataset"](params["path"], args.frames, args.stream)
    intrinsic = dataset.load_intrinsics(params["img_size"], params["input_size"])
    slam = build_slam(args, intrinsic, params)

    if args.stream:
        args.scene = "live"

    # NOTE: real-time semantic map construction
    if not os.path.exists(f"{args.data}_{args.scene}"):
        os.makedirs(f"{args.data}_{args.scene}")
    if args.load:
        if os.path.exists(f"{args.data}_{args.scene}/{args.algo}.npz"):
            print("[*] loading saved state...")
            slam.point_state.load(f"{args.data}_{args.scene}/{args.algo}.npz")
        else:
            print("[*] no saved state found, skipping...")
    else:
        if args.stream:
            stream_loop(args, slam)
        else:
            dataset_loop(args, slam, dataset)
            if args.save:
                slam.save(f"{args.data}_{args.scene}/{args.algo}.npz")

    # NOTE: save point cloud
    points, colors = slam.point_state.get_pc()
    save_pc(points, colors, f"{args.data}_{args.scene}/color_pc.ply")

    # NOTE: save colorized mesh
    mesh = slam.point_state.get_mesh()
    o3d.io.write_triangle_mesh(f"{args.data}_{args.scene}/color_mesh.ply", mesh)
    o3d.io.write_triangle_mesh(f"{args.data}_{args.scene}/color_mesh.glb", mesh)

    # NOTE: modify below to play with query
    if args.algo in ["cfusion", "vlfusion"]:
        # points, colors = slam.query("Window", topk=3)
        # points, colors = slam.query("there is a stainless steel fridge in the ketchen", topk=3)
        points, colors = slam.semantic_query([
            "vase", "table", "tv shelf", "curtain", "wall", "floor", "ceiling", "door", "tv",
            "room plant", "light", "sofa", "cushion", "wall paint", "chair"
        ])
        show_pc(points, colors, slam.point_state.poses)
        save_pc(points, colors, f"{args.data}_{args.scene}/semantic_pc.ply")

if __name__ == "__main__":
    main()