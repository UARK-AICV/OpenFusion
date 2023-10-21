import socket, pickle, struct
import threading, time
import open3d as o3d
import numpy as np
import pyrealsense2 as rs
from collections import namedtuple
from functools import partial
import queue
import bisect


class KinectCamera(object):
    def __init__(self, config=None, device=0, align_depth_to_color=True):
        if config is None:
            config = o3d.io.AzureKinectSensorConfig()
        else:
            config = o3d.io.read_azure_kinect_sensor_config(config)

        if device < 0 or device > 255:
            print('[*] Unsupported device id, fall back to 0')
            device = 0

        self.sensor = o3d.io.AzureKinectSensor(config)
        if not self.sensor.connect(device):
            raise RuntimeError('[*] Failed to connect to sensor')
        self.align_depth_to_color = align_depth_to_color

    def rec(self, record_enable, result_queue:queue.Queue):
        while record_enable.is_set():
            rgbd = self.sensor.capture_frame(self.align_depth_to_color)
            timestamp = time.time()
            if rgbd is None:
                continue
            # obrain (360, 640, 4) array
            rgb = np.asarray(rgbd.color)[::2, ::2]
            depth = np.asarray(rgbd.depth)[::2, ::2]
            result_queue.put_nowait((timestamp, rgb, depth))
            time.sleep(0.005)


class T265(object):
    def __init__(self, attrs=['rotation', 'translation']) -> None:
        ctx = rs.context()
        assert len(ctx.devices), "No Device detected."
        print(len(ctx.devices))
        self.attrs = attrs
        self.Pose = namedtuple('Pose', attrs)
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.pose)
        self.pipeline.start(cfg)

    def rec(self, record_enable, result_queue:queue.Queue):
        while record_enable.is_set():
            frames = self.pipeline.wait_for_frames()
            pose_frame = frames.get_pose_frame()

            if pose_frame:
                pose = pose_frame.get_pose_data()
                timestamp = time.time()
                p = self.Pose(*map(partial(getattr, pose), self.attrs))
                r, t = p.rotation, p.translation
                result_queue.put_nowait((timestamp, (t.x, t.y, t.z, r.x, r.y, r.z, r.w)))
                time.sleep(0.005)


class Server(object):
    """Camera server running on the robot

    Args:
        config (str): path to Kinect camera config file
    """
    def __init__(self, config=None, device=0, align_depth_to_color=True, with_pose=True, ts_thresh=0.01):
        self.ts_thresh = ts_thresh
        #! do not switch the order of these two recorders
        self.with_pose = with_pose
        if with_pose:
            self.pose_timestamps = []
            self.extrinsic_stream = []
            self.pose_queue = queue.Queue()
            self.t265 = T265()
        self.rgbd_queue = queue.Queue()
        self.azure_kinect = KinectCamera(config, device, align_depth_to_color)
        self.record_enable = threading.Event()
        self.sample_time = -1

    def __enter__(self):
        self.start()

    def __exit__(self):
        self.close()

    def start(self):
        self.record_enable.set()
        self.rgbd_record_process = threading.Thread(
            target=self.azure_kinect.rec,
            args=(
                self.record_enable,
                self.rgbd_queue,
            ),
            name="rgbd_record",
            daemon=True,
        )
        self.rgbd_record_process.start()
        if self.with_pose:
            self.pose_record_process = threading.Thread(
                target=self.t265.rec,
                args=(
                    self.record_enable,
                    self.pose_queue,
                ),
                name="pose_record",
                daemon=True,
            )
            self.pose_record_process.start()
        self.com_process = threading.Thread(
            target=self.get_server(port=8888),
            args=(
                (self.record_enable,)
            ),
            name="com",
            daemon=True,
        )
        self.com_process.start()

    def close(self):
        self.record_enable.clear()
        self.rgbd_record_process.join()
        if self.with_pose:
            self.pose_record_process.join()

    def get_server(self, port=9000):
        server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        host_name  = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        print(f'[*] HOST IP: {host_ip}')
        socket_address = (host_ip, port)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(socket_address)
        # Socket Listen
        server_socket.listen(5)

        def server(record_enable):
            while record_enable.is_set():
                print(f'[*] WAITING FOR CONNECTION AT: {socket_address}')
                client_socket, addr = server_socket.accept()
                print(f'[*] GOT CONNECTION FROM: {addr}')
                # don't use timeout and reconnect as it will mess up the pickle stream
                while client_socket:
                    res = self.get_inputs()
                    if res is None:
                        continue
                    data = pickle.dumps(res)
                    print(f"[*] Sending data with timestamp: {res['camera_time']}")
                    client_socket.send(struct.pack("Q", len(data)) + data) # float data
                client_socket.close()
        return server

    def state_dict(self):
        data = list(self.rgbd_queue.queue)
        self.rgbd_queue.queue.clear()
        res = {
            "camera_timestamps": [d[0] for d in data],
            "rgb_stream": [d[1] for d in data],
            "depth_stream": [d[2] for d in data],
        }
        if self.with_pose:
            data = list(self.pose_queue.queue)
            self.pose_queue.queue.clear()
            if len(data) == 0:
                return None
            res["pose_timestamps"] =  self.pose_timestamps + [d[0] for d in data]
            res["extrinsic_stream"] = self.extrinsic_stream + [d[1] for d in data]

            self.res = res["extrinsic_stream"][-1]
        return res

    def get_inputs(self):
        state = self.state_dict()
        if state is None:
            return None
        if len(state["camera_timestamps"]) == 0:
            return None

        cam_ts = state["camera_timestamps"][-1]
        if cam_ts == self.sample_time:
            return None

        res = {
            "camera_time": cam_ts,
            "rgb": state["rgb_stream"][-1],
            "depth": state["depth_stream"][-1],
        }
        if self.with_pose:
            if len(state["pose_timestamps"]) == 0:
                return None
            pose_idx = min(len(state["pose_timestamps"]) - 1, bisect.bisect_right(state["pose_timestamps"], cam_ts))
            if abs(state["pose_timestamps"][pose_idx] - cam_ts) < abs(state["pose_timestamps"][pose_idx-1] - cam_ts):
                if abs(state["pose_timestamps"][pose_idx] - cam_ts) > self.ts_thresh:
                    print(f"pose ts too far! pose:{state['pose_timestamps'][pose_idx]} and cam: {cam_ts}")
                    self.sample_time = cam_ts
                    return None
                res["pose_time"] = state["pose_timestamps"][pose_idx]
                res["extrinsic"] = state["extrinsic_stream"][pose_idx]
            else:
                if abs(state["pose_timestamps"][pose_idx-1] - cam_ts) > self.ts_thresh:
                    print(f"pose ts too far! pose:{state['pose_timestamps'][pose_idx]} and cam: {cam_ts}")
                    self.sample_time = cam_ts
                    return None
                res["pose_time"] = state["pose_timestamps"][pose_idx-1]
                res["extrinsic"] = state["extrinsic_stream"][pose_idx-1]

        if self.with_pose:
            self.extrinsic_stream = state["extrinsic_stream"][pose_idx+1:]
            self.pose_timestamps = state["pose_timestamps"][pose_idx+1:]
        self.sample_time = cam_ts
        del state
        return res


def main():
    server = Server(config="deploy/default_config.json")
    try:
        server.start()
        while True:
            time.sleep(0.1)
    except:
        server.close()


if __name__ == "__main__":
    main()