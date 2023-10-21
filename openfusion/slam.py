import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.dlpack
import open3d as o3d
import open3d.core as o3c
import time
import threading
import os
from openfusion.io import BaseIO, SocketIO
from openfusion.state import BaseState, VLState, CFState
from openfusion.zoo.base_model import VLFM, build_vl_model
from openfusion.utils import rand_cmap, get_cmap_legend
from matplotlib import pyplot as plt
from typing import List, Tuple, Union, Optional, Dict, Any
try:
    import torchshow as ts
except:
    print("[*] install `torchshow` for easier debugging")


class BaseSLAM(object):
    def __init__(
        self,
        intrinsic,
        io: BaseIO,
        point_state: BaseState,
        with_pose=True,
        img_size=(640, 480),
        live_mode=False,
        capture=False
    ) -> None:
        self.intrinsic = intrinsic
        self.io = io
        self.point_state = point_state
        self.with_pose = with_pose
        self.height = img_size[1]
        self.width = img_size[0]
        self.export_path = None
        self.live_mode = live_mode
        self.capture = capture
        self.selected_points = None
        self.selected_colors = None
        self.control_thread_enabled = False
        self.monitor_thread_enabled = False

    def save(self, path=None):
        if path is None:
            path = self.export_path
        self.point_state.save(path)

    def load(self, path):
        self.point_state.load(path)

    def vo(self):
        inputs = self.io.get_inputs()
        if inputs is None:
            return

        rgb = np.ascontiguousarray(inputs["rgb"])
        depth = np.ascontiguousarray(inputs["depth"])
        depth[depth <= 0] = 0

        # NOTE: we require camera pose for the current implementation
        if self.with_pose:
            extrinsic = inputs["extrinsic"]
            self.point_state.append(rgb, depth, extrinsic)
        else:
            raise NotImplementedError("[*] add any vo method here")

    def compute_state(self, bs=1, **kwargs):
        if len(self.point_state.rgb_buffer) < bs:
            return
        batch_color, batch_depth, batch_extrinsic = self.point_state.get(bs)

        for color, depth, extrinsic in zip(batch_color, batch_depth, batch_extrinsic):
            self.point_state.update(color, depth, extrinsic)

    def start_thread(self):
        self.control_thread_enabled = True
        self.control_thread = threading.Thread(
            target=self.get_control_function(self.vo, control_interval=0.01) # ASAP
        )
        self.control_thread.start()
        self.state_thread = threading.Thread(
            target=self.get_control_function(self.compute_state, control_interval=0.01),
            args=(1,)
        )
        self.state_thread.start()
        print("[*] Control thread started")
        if self.live_mode:
            self.start_monitor_thread()

    def start_monitor_thread(self):
        time.sleep(1.0)
        self.monitor_thread_enabled = True
        self.monitor_thread = threading.Thread(
            target=self.get_monitor_function(self.monitor, control_interval=0.1)
        )
        self.monitor_thread.start()
        print("[*] Monitor thread started")

    def stop_thread(self):
        self.control_thread_enabled = False
        self.control_thread.join()
        self.state_thread.join()
        if self.export_path is not None:
            self.save()
        print("[*] Control thread stopped")
        if self.live_mode:
            self.stop_monitor_thread()

    def stop_monitor_thread(self):
        self.monitor_thread_enabled = False
        self.monitor_thread.join()
        print("[*] Monitor thread stopped")

    def get_control_function(self, function, control_interval):
        def control_loop(*args, **kwargs):
            setattr(self, f"{function.__name__}_step", 0)
            setattr(self, f"{function.__name__}_time", time.time())
            while self.control_thread_enabled:
                start_time = time.time()
                function(*args, **kwargs)
                end_time = time.time()
                time.sleep(max(0, control_interval - (end_time - start_time)))
                self.__dict__[f"{function.__name__}_step"] += 1
                self.__dict__[f"{function.__name__}_time"] = start_time
        return control_loop

    def get_monitor_function(self, function, control_interval):
        def control_loop():
            setattr(self, f"{function.__name__}_step", 0)
            # NOTE: reate a visualizer
            self.vis = o3d.visualization.Visualizer()
            self.pcd = o3d.geometry.PointCloud()
            self.selected_pcd = o3d.geometry.PointCloud()
            self.cur_camera_cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            self.cur_camera_lines = o3d.geometry.LineSet.create_camera_visualization(
                view_width_px=self.width, view_height_px=self.height,
                intrinsic=self.intrinsic,
                extrinsic=np.eye(4)
            )
            # NOTE: initialize visualizer window
            print("[*] Initializing visualizer window...")
            self.vis.create_window(width=1200, height=680)
            self.is_monitor_init = False
            if self.capture:
                if not os.path.exists("capture"):
                    os.mkdir("capture")
            while self.monitor_thread_enabled:
                start_time = time.time()
                function()
                end_time = time.time()
                time.sleep(max(0, control_interval - (end_time - start_time)))
                if self.capture:
                    self.vis.capture_screen_image(
                        f"capture/rgb_{self.__dict__[f'{function.__name__}_step']:04d}.jpg",
                        do_render=True
                    )
                self.__dict__[f"{function.__name__}_step"] += 1
            self.vis.destroy_window()
        return control_loop

    def monitor(self):
        if len(self.point_state.poses) <= 1:
            return
        time.sleep(0.02)
        points, colors = self.point_state.get_pc(500000)
        if len(points) == 0:
            return
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        if self.selected_points is not None:
            # NOTE: highlight selected points
            self.selected_pcd.points = o3d.utility.Vector3dVector(self.selected_points)
            self.selected_pcd.colors = o3d.utility.Vector3dVector(self.selected_colors)
        pose = np.linalg.inv(self.point_state.get_last_pose())

        if not self.is_monitor_init:
            self.vis.add_geometry(self.pcd)
            self.vis.add_geometry(self.selected_pcd)
            self.last_pose = pose
            self.cur_camera_cf.transform(pose)
            self.cur_camera_lines.transform(pose)
            self.vis.add_geometry(self.cur_camera_cf)
            self.vis.add_geometry(self.cur_camera_lines)
            self.is_monitor_init = True
        else:
            self.vis.update_geometry(self.pcd)
            self.vis.update_geometry(self.selected_pcd)

            if np.any(pose != self.last_pose):
                transform = pose @ np.linalg.inv(self.last_pose)
                self.cur_camera_cf.transform(transform)
                self.cur_camera_lines.transform(transform)
                self.vis.update_geometry(self.cur_camera_cf)
                self.vis.update_geometry(self.cur_camera_lines)
                self.last_pose = pose

        # Force the visualizer to redraw
        self.vis.poll_events()
        self.vis.update_renderer()


class VLSLAM(BaseSLAM):
    def __init__(
        self,
        intrinsic,
        io,
        point_state:BaseState,
        with_pose=True,
        img_size=(640, 480),
        vl_model:VLFM=None,
        host_ip="127.0.0.1",
        query_port=5000,
        live_mode=False
    ) -> None:
        super().__init__(intrinsic, io, point_state, with_pose, img_size, live_mode)
        if isinstance(point_state, CFState):
            self.mode = "emb"
        elif isinstance(point_state, VLState):
            self.mode = "default"
        else:
            self.mode = "default"
        self.vl_model = vl_model
        self.host_ip = host_ip
        self.query_port = query_port

    @torch.no_grad()
    def compute_state(self, bs=1, encode_image=True):
        if len(self.point_state.rgb_buffer) < bs:
            return
        batch_color, batch_depth, batch_extrinsic = self.point_state.get(bs)

        if self.control_thread_enabled:
            if self.compute_state_step % 15 == 0:
                res_list = self.vl_model.encode_image(batch_color, mode=self.mode)
            else:
                res_list = [{} for _ in range(bs)]
        else:
            if encode_image:
                res_list = self.vl_model.encode_image(batch_color, mode=self.mode)
            else:
                res_list = [{} for _ in range(bs)]

        for color, depth, extrinsic, res_dict in zip(
            batch_color, batch_depth, batch_extrinsic, res_list
        ):
            self.point_state.update(color, depth, extrinsic, res_dict)

    def start_query_thread(self, fast_query=False):
        self.query_thread_enabled = True
        self.query_thread = threading.Thread(
            target=self.get_query_function(
                self.fast_query if fast_query else self.query,
                control_interval=1
            )
        )
        self.query_thread.start()
        print("Query thread started")

    def stop_query_thread(self):
        self.query_thread_enabled = False
        self.query_thread.join()
        print("Query thread stopped")

    def get_query_function(self, f_query, control_interval):
        import socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_socket.bind((self.host_ip, self.query_port))
        def query_server():
            while self.query_thread_enabled:
                data, addr = server_socket.recvfrom(1024)
                data = data.decode("utf-8")
                print(f"[*] query from {str(addr)}: {data}")
                self.selected_points = f_query(data, only_poi=True)
                if self.selected_points is not None:
                    self.selected_colors = np.array([1, 0, 0]) * np.ones_like(self.selected_points)
                time.sleep(control_interval)
        return query_server

    @torch.no_grad()
    def fast_query(
        self, query="chair", points=None, colors=None, only_poi=False, topk=1, n_points=500000
    ):
        """ perform sparse object query on the point cloud

        Args:
            query (List[str]): strings of semantic classes
            points (np.array, optional): points to query. Defaults to None.
            colors (np.array, optional): original color of points. Defaults to None.
            only_poi (bool, optional): return only points of inretest. Defaults to False.
            topk (int, optional): top k elements to consider. Defaults to 1.
            n_points (int, optional): number of points to query. -1 to select all. Defaults to 500000.

        Returns:
            points (np.array): xyz coordinates
            colors (np.array): colored points
        """
        t_emb = self.vl_model.encode_prompt(query, task="default")
        if points is None or colors is None:
            assert points is None and colors is None, "[*] points and colors should be both None or provided"
            points, colors = self.point_state.get_pc(n_points)
        return self.point_state.fast_object_query(t_emb, points, colors, only_poi=only_poi, topk=topk)

    @torch.no_grad()
    def query(
        self, query="chair", points=None, colors=None, only_poi=False, topk=1, n_points=500000, **kwargs
    ):
        """ perform dense object query on the point cloud

        Args:
            query (List[str]): strings of semantic classes
            points (np.array, optional): points to query. Defaults to None.
            colors (np.array, optional): original color of points. Defaults to None.
            only_poi (bool, optional): return only points of inretest. Defaults to False.
            topk (int, optional): top k elements to consider. Defaults to 1.
            n_points (int, optional): number of points to query. -1 to select all. Defaults to 500000.

        Returns:
            points (np.array): xyz coordinates
            colors (np.array): colored points
        """
        t_emb = self.vl_model.encode_prompt(query, task="default")
        if points is None or colors is None:
            assert points is None and colors is None, "[*] points and colors should be both None or provided"
            points, colors = self.point_state.get_pc(n_points)
        return self.point_state.object_query(t_emb, points, colors, only_poi=only_poi, topk=topk, **kwargs)

    @torch.no_grad()
    def semantic_query(
        self, query:List[str], points:np.array=None, colors:np.array=None, cmap=None, n_points=500000
    ):
        """ perform semantic segmentation on the point cloud
        Args:
            query (List[str]): strings of semantic classes
            points (np.array, optional): points to query. Defaults to None.
            colors (np.array, optional): original color of points. Defaults to None.
            cmap (_type_, optional): colormap to use. Defaults to None.
            n_points (int, optional): number of points to query. -1 to select all. Defaults to 500000.

        Returns:
            points (np.array): xyz coordinates
            colors (np.array): colored points
        """
        t_emb = self.vl_model.encode_prompt(query, task="default")
        if points is None or colors is None:
            assert points is None and colors is None, "[*] points and colors should be both None or provided"
            points, colors = self.point_state.get_pc(n_points)
        if cmap is None:
            cmap = rand_cmap(len(query), type="bright", first_color_black=False)
            get_cmap_legend(cmap, query)
        return self.point_state.semantic_query(t_emb, points, colors, cmap)

    @torch.no_grad()
    def panoptic_query(
        self, query:List[str], points:np.array=None, colors:np.array=None, cmap=None, metadata=None, n_points=500000
    ):
        """ perform panoptic segmentation on the point cloud

        Args:
            query (List[str]): strings of semantic classes
            points (np.array, optional): points to query. Defaults to None.
            colors (np.array, optional): original color of points. Defaults to None.
            cmap (_type_, optional): colormap to use. Defaults to None.
            metadata (_type_, optional): metadata used for panoptic post_process. Defaults to None.
            n_points (int, optional): number of points to query. -1 to select all. Defaults to 500000.

        Returns:
            points (np.array): xyz coordinates
            colors (np.array): colored points
        """
        t_emb = self.vl_model.encode_prompt(query, task="default")
        if points is None or colors is None:
            assert points is None and colors is None, "[*] points and colors should be both None or provided"
            points, colors = self.point_state.get_pc(n_points)
        return self.point_state.panoptic_query(t_emb, points, colors, cmap, metadata)

    def mesh_fast_query(self, query="chair", **kwargs):
        mesh = self.point_state.get_mesh()
        points = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors)
        points, colors = self.fast_query(query, points=points, colors=colors, **kwargs)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        return mesh

    def mesh_query(self, query="chair", **kwargs):
        mesh = self.point_state.get_mesh()
        points = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors)
        points, colors = self.query(query, points=points, colors=colors, **kwargs)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        return mesh

    def mesh_semantic_query(self, query="chair", **kwargs):
        mesh = self.point_state.get_mesh()
        points = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors)
        points, colors = self.semantic_query(query, points=points, colors=colors, **kwargs)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        return mesh

    def mesh_panoptic_query(self, query="chair", **kwargs):
        mesh = self.point_state.get_mesh()
        points = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors)
        points, colors = self.panoptic_query(query, points=points, colors=colors, **kwargs)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        return mesh


def build_slam(args, intrinsic, params):
    with_pose = True
    if args.stream:
        io = SocketIO(with_pose, thresh=0.01, host_ip=args.host_ip)
    else:
        assert params["img_size"][0] / params["input_size"][0] == \
               params["img_size"][1] / params["input_size"][1], \
               "[*] img_size and input_size should have the same aspect ratio"
        assert params["img_size"][0] % params["input_size"][0] == 0, \
                "[*] img_size should be divisible by input_size"
        io = BaseIO(
            with_pose, thresh=0.01,
            inverse_scale=int(params["img_size"][0] / params["input_size"][0])
        )

    if args.algo == "default":
        point_state = BaseState(
            intrinsic, params["depth_scale"], params["depth_max"],
            params["voxel_size"], params["block_resolution"], params["block_count"],
            device=args.device.upper(), img_size=params["input_size"]
        )
        return BaseSLAM(intrinsic, io, point_state, with_pose, params["input_size"], live_mode=False)
    elif args.algo == "cfusion":
        vl_model = build_vl_model(args.vl, input_size=min(360, params["input_size"][1]))
        point_state = CFState(
            intrinsic, params["depth_scale"], params["depth_max"],
            params["voxel_size"], params["block_resolution"], params["block_count"],
            dim=vl_model.dim, device=args.device.upper(), img_size=params["input_size"]
        )
        return VLSLAM(intrinsic, io, point_state, with_pose, params["input_size"], vl_model, live_mode=False)
    elif args.algo == "vlfusion":
        from .matcher import HungarianMatcher
        vl_model = build_vl_model(args.vl, input_size=min(360, params["input_size"][1]))
        point_state = VLState(
            intrinsic, params["depth_scale"], params["depth_max"],
            params["voxel_size"], params["block_resolution"], params["block_count"],
            device=args.device.upper(), img_size=params["input_size"],
            matcher=HungarianMatcher(num_points=3600)
        )
        return VLSLAM(intrinsic, io, point_state, with_pose, params["input_size"], vl_model)
    else:
        raise ValueError("Unknown SLAM algorithm: {}".format(args.algo))
