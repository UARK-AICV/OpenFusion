import argparse
import os
import os
import struct
import numpy as np
import zlib
import imageio
import cv2
from PIL import Image
import tqdm


COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {
    -1: "unknown",
    0: "raw_ushort",
    1: "zlib_ushort",
    2: "occi_ushort",
}


class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

    def decompress_depth(self, compression_type):
        if compression_type == "zlib_ushort":
            return self.decompress_depth_zlib()
        else:
            raise NotImplementedError

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == "jpeg":
            return self.decompress_color_jpeg()
        else:
            raise NotImplementedError

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)


class SensorData:
    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = f.read(strlen).decode("utf-8")
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[
                struct.unpack("i", f.read(4))[0]
            ]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[
                struct.unpack("i", f.read(4))[0]
            ]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export_depth_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(
            f"[*] exporting {len(self.frames) // frame_skip} depth frames to {output_path}"
        )
        for f in tqdm.tqdm(range(0, len(self.frames), frame_skip)):
            depth_data = self.frames[f].decompress_depth(
                self.depth_compression_type
            )
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                self.depth_height, self.depth_width
            )
            if image_size is not None:
                depth = cv2.resize(
                    depth,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            Image.fromarray(depth.astype(np.uint16)).save(os.path.join(output_path, str(f) + ".png"))

    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f'[*] exporting {len(self.frames) // frame_skip} color frames to {output_path}')
        for f in tqdm.tqdm(range(0, len(self.frames), frame_skip)):
            color = self.frames[f].decompress_color(
                self.color_compression_type
            )
            if image_size is not None:
                color = cv2.resize(
                    color,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            imageio.imwrite(os.path.join(output_path, str(f) + ".png"), color)

    @staticmethod
    def save_mat_to_file(matrix, filename):
        with open(filename, "w") as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt="%f")

    def export_poses(self, output_path, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f'[*] exporting {len(self.frames) // frame_skip} camera poses to {output_path}')
        for f in tqdm.tqdm(range(0, len(self.frames), frame_skip)):
            self.save_mat_to_file(
                self.frames[f].camera_to_world,
                os.path.join(output_path, str(f) + ".txt"),
            )

    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(f'[*] exporting camera intrinsics to {output_path}')
        self.save_mat_to_file(
            self.intrinsic_color,
            os.path.join(output_path, "intrinsic_color.txt"),
        )
        self.save_mat_to_file(
            self.extrinsic_color,
            os.path.join(output_path, "extrinsic_color.txt"),
        )
        self.save_mat_to_file(
            self.intrinsic_depth,
            os.path.join(output_path, "intrinsic_depth.txt"),
        )
        self.save_mat_to_file(
            self.extrinsic_depth,
            os.path.join(output_path, "extrinsic_depth.txt"),
        )


def main(opt):
    opt.output_path = os.path.join(opt.output_path, os.path.basename(opt.filename).split('.')[0])
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    print(f'[*] loading {opt.filename}...')
    sd = SensorData(opt.filename)
    print('[*] loaded!')
    if opt.export_depth_images:
        sd.export_depth_images(os.path.join(opt.output_path, 'depth'))
    if opt.export_color_images:
        sd.export_color_images(os.path.join(opt.output_path, 'rgb'))
    if opt.export_poses:
        sd.export_poses(os.path.join(opt.output_path, 'pose'))
    if opt.export_intrinsics:
        sd.export_intrinsics(os.path.join(opt.output_path, 'intrinsic'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f', help='path to .sens file to read')
    parser.add_argument('--output_path', default=".", help='path to output folder')
    parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
    parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
    parser.add_argument('--export_poses', dest='export_poses',  action='store_true')
    parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')

    opt = parser.parse_args()
    main(opt)