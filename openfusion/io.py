import time
import bisect
import numpy as np
from PIL import Image
import socket, pickle, struct
import queue
import threading
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot
from openfusion.utils import relative_transform, kobuki_pose2rgbd


class BaseIO(object):
    def __init__(self, with_pose=False, thresh=0.01, inverse_scale=1) -> None:
        self.rgb_stream = []
        self.depth_stream = []
        self.camera_timestamps = []
        self.with_pose = with_pose
        if with_pose:
            self.extrinsic_stream = []
            self.pose_timestamps = []
            self.ts_thresh = thresh
        self.sample_time = -1
        self.inverse_scale = inverse_scale
        assert inverse_scale in [1, 2], "[*] inverse_scale must be 1 or 2"

    def from_file(self, rgb_path, depth_path):
        rgb = np.array(Image.open(rgb_path), dtype=np.uint8)
        depth = np.array(Image.open(depth_path), dtype=np.uint16)
        if self.inverse_scale != 1:
            rgb = rgb[::2, ::2]
            depth = depth[::2, ::2]
        if rgb.shape != depth.shape:
            rgb = np.array(Image.fromarray(rgb).resize((depth.shape[1], depth.shape[0])), dtype=np.uint8)
        return rgb, depth

    def update(self, rgb=None, depth=None, pose=None):
        if rgb is not None and depth is not None:
            self.rgb_stream.append(rgb)
            self.depth_stream.append(depth)
            self.camera_timestamps.append(time.time())
        if pose is not None:
            assert self.with_pose
            self.pose_timestamps.append(time.time())
            self.extrinsic_stream.append(pose)

    def process_pose(self, pose_quat):
        return pose_quat

    def state_dict(self):
        res = {
            "rgb_stream": self.rgb_stream.copy(),
            "depth_stream": self.depth_stream.copy(),
            "camera_timestamps": self.camera_timestamps.copy()
        }
        if self.with_pose:
            res["pose_timestamps"] = self.pose_timestamps.copy()
            res["extrinsic_stream"] = self.extrinsic_stream.copy()
        return res

    def get_inputs(self):
        last_time = self.sample_time
        # get elements newer than last_time
        state = self.state_dict()
        if len(state["camera_timestamps"]) == 0:
            return None

        start_idx = min(len(state["camera_timestamps"]), bisect.bisect_right(state["camera_timestamps"], last_time))
        cam_ts = state["camera_timestamps"][start_idx]

        if cam_ts == self.sample_time:
            return None

        res = {
            "rgb": state["rgb_stream"][start_idx],
            "depth": state["depth_stream"][start_idx],
            "camera_time": cam_ts
        }
        if self.with_pose:
            pose_idx = min(len(state["pose_timestamps"]), bisect.bisect_right(state["pose_timestamps"], cam_ts))
            if abs(state["pose_timestamps"][pose_idx] - cam_ts) < abs(state["pose_timestamps"][pose_idx-1] - cam_ts):
                if abs(state["pose_timestamps"][pose_idx] - cam_ts) > self.ts_thresh:
                    return None
                res["pose_time"] = state["pose_timestamps"][pose_idx]
                res["extrinsic"] = self.process_pose(state["extrinsic_stream"][pose_idx])
            else:
                if abs(state["pose_timestamps"][pose_idx-1] - cam_ts) > self.ts_thresh:
                    return None
                res["pose_time"] = state["pose_timestamps"][pose_idx-1]
                res["extrinsic"] = self.process_pose(state["extrinsic_stream"][pose_idx-1])

        self.rgb_stream = state["rgb_stream"][start_idx+1:]
        self.depth_stream = state["depth_stream"][start_idx+1:]
        self.camera_timestamps = state["camera_timestamps"][start_idx+1:]
        if self.with_pose:
            self.extrinsic_stream = state["extrinsic_stream"][pose_idx+1:]
            self.pose_timestamps = state["pose_timestamps"][pose_idx+1:]
        self.sample_time = cam_ts
        del state
        return res


class SocketIO(BaseIO):
    def __init__(
        self, with_pose=False, thresh=0.01, inverse_scale=1,
        host_ip="127.0.0.1", com_port=8888
    ) -> None:
        super().__init__(with_pose, thresh, inverse_scale)
        self.host_ip = host_ip
        self.record_enable = threading.Event()
        self.result_queue = queue.Queue()
        self.com_process = threading.Thread(
            target=self.get_client(com_port, self.result_queue),
            args=(self.record_enable,),
            name="com_process",
            daemon=True,
        )
        self.record_enable.set()
        self.com_process.start()
        print("[*] Com thread started")

        if with_pose:
            self.initial_pose = None
            self.pose2rgbd = kobuki_pose2rgbd()

    def close(self):
        self.record_enable.clear()
        self.com_process.join()

    def get_client(self, port, result_queue:queue.Queue):
        print("[*] Setting up client...")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.host_ip, port))
        print(f"[*] Client connected to ('{self.host_ip}', {port})")
        payload_size = struct.calcsize("Q")
        def recvall(sock, count):
            buf = b''
            while count:
                newbuf = sock.recv(count)
                if not newbuf:
                    return None
                buf += newbuf
                count -= len(newbuf)
            return buf

        def client(record_enable):
            while record_enable.is_set():
                length = recvall(client_socket, payload_size)
                data = recvall(client_socket, int.from_bytes(length, "little"))
                result_queue.put_nowait(pickle.loads(data))
            client_socket.close()
        return client

    def get_inputs(self):
        last_time = self.sample_time
        if self.result_queue.empty():
            return None

        while not self.result_queue.empty():
            res = self.result_queue.get_nowait()
            if res["camera_time"] < last_time:
                continue
            if self.with_pose:
                res["extrinsic"] = self.process_pose(res["extrinsic"])
            self.sample_time = res["camera_time"]
            return res

    def process_pose(self, pose_quat):
        r = Rot.from_quat(pose_quat[3:]).as_matrix()
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = r
        pose_matrix[:3, 3] = pose_quat[:3]
        if self.initial_pose is None:
            self.initial_pose = pose_matrix
        return np.linalg.inv(
            relative_transform(self.initial_pose, pose_matrix).astype(np.float64) @ self.pose2rgbd
        )


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation
    from datetime import datetime
    import open3d as o3d

    SAVE = True
    HOST_IP = "10.35.182.224"
    io = SocketIO(with_pose=True, host_ip=HOST_IP, com_port=8888)

    ax1 = plt.subplot(1,1,1)
    im1 = ax1.imshow(np.zeros((360, 640, 3)))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    camera_cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(camera_cf)
    last_pose = None

    if SAVE:
        save_rgb, save_depth, save_pose = [], [], []

    def update(i):
        global last_pose
        inputs = io.get_inputs()
        if inputs is None:
            return
        im1.set_data(inputs["rgb"])
        print(inputs["camera_time"] - inputs["pose_time"])

        if last_pose is None:
            camera_cf.transform(inputs["extrinsic"])
            last_pose = inputs["extrinsic"]
            vis.update_geometry(camera_cf)

        else:
            transform = inputs["extrinsic"] @ np.linalg.inv(last_pose)
            camera_cf.transform(transform)
            last_pose = inputs["extrinsic"]
            vis.update_geometry(camera_cf)

        vis.poll_events()
        vis.update_renderer()

        if SAVE:
            save_rgb.append(inputs["rgb"])
            save_depth.append(inputs["depth"])
            save_pose.append(inputs["extrinsic"])

    try:
        ani = FuncAnimation(plt.gcf(), update, interval=10)
        plt.show()
    finally:
        if SAVE:
            scene = datetime.now().strftime("%Y%m%d_%H%M%S")

            if not os.path.exists(f"sample/live/{scene}"):
                os.makedirs(f"sample/live/{scene}/rgb")
                os.makedirs(f"sample/live/{scene}/depth")
                os.makedirs(f"sample/live/{scene}/pose")
            print("[*] Saving {} frames".format(len(save_rgb)))
            for i, (rgb, depth, pose) in tqdm(enumerate(zip(save_rgb, save_depth, save_pose))):
                Image.fromarray(rgb.astype(np.uint8)).save(f"sample/live/{scene}/rgb/{i}.png")
                Image.fromarray(depth.astype(np.uint16)).save(f"sample/live/{scene}/depth/{i}.png")
                np.savetxt(f"sample/live/{scene}/pose/{i}.txt", pose)
        io.close()