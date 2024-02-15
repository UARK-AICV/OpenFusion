import numpy as np
import torch
from torch.nn import functional as F
import torch.utils.dlpack
import open3d as o3d
import open3d.core as o3c
import matplotlib.pyplot as plt
import bisect
from sklearn.neighbors import KDTree
try:
    import torchshow as ts
    import time
    from openfusion.utils import rand_cmap
    DBG = False
except:
    print("[*] torchshow not found")


class BaseState(object):
    def __init__(
        self,
        intrinsic,
        depth_scale,
        depth_max,
        voxel_size = 5.0 / 512,
        block_resolution = 8,
        block_count = 100000,
        device = "CUDA:0",
        img_size=(640, 480)
    ) -> None:
        self.timestamp = None
        self.img_size = img_size
        self.device = o3c.Device(device)
        self.depth_scale = depth_scale
        self.depth_max = depth_max
        self.voxel_size = voxel_size
        self.trunc = self.voxel_size * 4
        self.block_resolution = block_resolution
        self.intrinsic_np = intrinsic
        self.intrinsic = o3c.Tensor.from_numpy(intrinsic)
        self.world = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight', 'color'),
            (o3c.float32, o3c.float32, o3c.float32),
            ((1), (1), (3)),
            self.voxel_size,
            block_resolution,
            block_count, device=self.device
        )
        self.rgb_buffer = []
        self.depth_buffer = []
        self.poses_buffer = []
        self.poses = []

    def custom_intrinsic(self, w, h):
        """ rescales intrinsic matrix according to new image size
        Args:
            w (int): new width
            h (int): new height
        """
        intrinsic = self.intrinsic_np.copy()
        w0, h0 = self.img_size
        intrinsic[0] *= (w / w0)
        intrinsic[1] *= (h / h0)
        return o3c.Tensor.from_numpy(intrinsic)

    def save(self, path):
        self.world.save(path)
        data = np.load(path)
        np.savez(
            path,
            intrinsic = self.intrinsic_np,
            extrinsic = np.array(self.poses),
            **data
        )

    def load(self, path):
        self.world = self.world.load(path)
        data = np.load(path)
        self.intrinsic_np = data["intrinsic"]
        self.intrinsic = o3c.Tensor.from_numpy(self.intrinsic_np)
        self.poses = data["extrinsic"].tolist()

    def append(self, rgb, depth, extrinsic):
        self.rgb_buffer.append(rgb)
        self.depth_buffer.append(depth)
        self.poses_buffer.append(extrinsic)

    def get(self, bs=1):
        if len(self.rgb_buffer) < bs:
            return None, None, None
        if bs == 1:
            pose = self.poses_buffer.pop(0)
            self.poses.append(pose)
            return [self.rgb_buffer.pop(0),], [self.depth_buffer.pop(0),], [pose,]
        if bs > len(self.rgb_buffer):
            bs = len(self.rgb_buffer)
        rgb = [self.rgb_buffer.pop(0) for _ in range(bs)]
        depth = [self.depth_buffer.pop(0) for _ in range(bs)]
        poses = [self.poses_buffer.pop(0) for _ in range(bs)]
        self.poses.extend(poses)
        return rgb, depth, poses

    def get_last_pose(self):
        return self.poses[-1]

    def get_mesh(self, legacy=True):
        mesh = self.world.extract_triangle_mesh()
        return mesh.to_legacy() if legacy else mesh

    def get_pc(self, n=-1):
        if len(self.poses) < 1:
            return None, None
        pcd = self.world.extract_point_cloud()
        points = pcd.point.positions.cpu().numpy()
        colors = pcd.point.colors.cpu().numpy()
        if n > 0 and len(points) > n:
            sample_idx = np.random.choice(len(points), n)
            points = points[sample_idx]
            colors = colors[sample_idx]
        return points, colors

    def get_og2d(self, robot_height=0.72, camera_height=0.38, grid_size=0.02):
        """get 2D occupancy grid of the world
        Args:
            robot_height (float, optional): clearance to be considered in [m]. Defaults to 0.75.
            camera_height (float, optional): height of camera from ground in [m]. Defaults to 0.45.
        """
        pcd = self.world.extract_point_cloud().to_legacy()
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=grid_size)
        ux, _, uz = pcd.get_max_bound()
        lx, _, lz = pcd.get_min_bound()
        x_ = np.arange(lx, ux, 0.1)
        y_ = np.arange(camera_height-robot_height, camera_height-0.1, 0.05)
        z_ = np.arange(lz, uz, 0.1)
        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
        queries = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        output = np.array(voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))).reshape(x.shape)
        return np.any(output, axis=1), ((lx, ux), (lz, uz))

    def get_pos_in_og2d(self, lims=((0,0),(0,0)), pose=None, pos=None):
        """get 2d position in occupancy grid
        Args:
            lims (tuple, optional): limits of the occupancy grid (returned by get_og2d).
            pose (list, optional): camera pose
            pos (list, optional): point in world frame
        """
        if pos is None:
            if pose is None:
                pose = self.get_last_pose()
            pos = np.linalg.inv(pose)
        x, z = pos[0][3], pos[2][3]
        x_ = np.arange(*lims[0], 0.1)
        z_ = np.arange(*lims[1], 0.1)
        return (
            bisect.bisect_right(z_, z),
            bisect.bisect_right(x_, x),
        )

    @staticmethod
    def depth_to_point_cloud(depth, extrinsic, intrinsic, image_width, image_height, depth_max, depth_scale):
        """
        Args:
            depth (np.array): depth image
            extrinsic (o3c.Tensor): shape of (4, 4)
            intrinsic (o3c.Tensor): shape of (3, 3). Use self.custom_intrinsic(image_width, image_height)
            image_width (int): image width
            image_height (int): image height
            depth_max (float): depth max
            depth_scale (float): depth scale
        Returns:
            coords (torch.Tensor): shape of (N, 3)
            mask (torch.Tensor): shape of (H, W)
        """
        depth = torch.from_numpy(depth.astype(np.int32)) / depth_scale
        depth = F.interpolate(
            depth.unsqueeze(0).unsqueeze(0).float(),
            (image_height, image_width)
        ).view(image_height, image_width).cuda()
        extrinsic = torch.utils.dlpack.from_dlpack(extrinsic.to_dlpack()).cuda().float()
        intrinsic = torch.utils.dlpack.from_dlpack(intrinsic.to_dlpack()).cuda().float()
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]

        v, u = torch.meshgrid(torch.arange(image_height).cuda(), torch.arange(image_width).cuda(), indexing="ij")
        uvd = torch.stack([u, v, torch.ones_like(depth)], dim=0).float() # (3,H,W)
        # NOTE: don't use torch.inverse(intrinsic) as it is slow
        uvd[0] = (uvd[0] - cx) / fx
        uvd[1] = (uvd[1] - cy) / fy
        xyz = uvd.view(3, -1) * depth.view(1, -1) # (3, H*W)

        # NOTE: convert to world frame
        R = extrinsic[:3, :3].T
        coords =  (R @ xyz - R @ extrinsic[:3, 3:]).view(3, image_height, image_width).permute(1,2,0)
        mask = [(0 < depth) & (depth < depth_max)]
        # TODO: check 0.05 offset for +y direction (up)
        return coords[mask] + torch.tensor([[0,0.05,0]], device="cuda"), mask

    @staticmethod
    def get_points_in_fov(coords, extrinsic, intrinsic, image_width, image_height, depth_max):
        """
        Args:
            coords (o3c.Tensor): shape of (N, 3)
            extrinsic (o3c.Tensor): shape of (4, 4)
            intrinsic (o3c.Tensor): shape of (3, 3). Use self.custom_intrinsic(image_width, image_height)
            image_width (int): width of the image
            image_height (int): height of the image
            depth_max (float): depth max
        Returns:
            v_proj (torch.Tensor): shape of (M)
            u_proj (torch.Tensor): shape of (M)
            d_proj (torch.Tensor): shape of (M)
            mask_proj (torch.Tensor): shape of (N)
        """
        coords = torch.utils.dlpack.from_dlpack(coords.to_dlpack()).cuda().float()
        extrinsic = torch.utils.dlpack.from_dlpack(extrinsic.to_dlpack()).cuda().float()
        intrinsic = torch.utils.dlpack.from_dlpack(intrinsic.to_dlpack()).cuda().float()

        # NOTE: apply camera pose
        xyz = extrinsic[:3, :3] @ coords.T + extrinsic[:3, 3:]
        # NOTE: perform projection using the camera intrinsic matrix (W,H,D)
        uvd = intrinsic @ xyz
        d = uvd[2]
        # NOTE: divide by third coordinate to obtain 2D pixel locations
        u = (uvd[0] / d).long() # W
        v = (uvd[1] / d).long() # H

        # NOTE: filter out points outside the image plane (outside FoV)
        mask_proj = (depth_max > d) & (
            (d > 0) &
            (u >= 0) &
            (v >= 0) &
            (u < image_width) &
            (v < image_height)
        )
        v_proj = v[mask_proj] # H
        u_proj = u[mask_proj] # W
        d_proj = d[mask_proj] # D

        return v_proj, u_proj, d_proj, mask_proj

    def active_buf_indices(self):
        # Find all active buf indices in the underlying engine
        buf_indices = self.world.hashmap().active_buf_indices()
        return buf_indices

    def active_buf_indices_in_fov(self, extrinsic, width, height, device):
        pcd = self.world.extract_point_cloud()
        points = pcd.point.positions
        _, _, _, mask_proj = self.get_points_in_fov(
            points, extrinsic, self.custom_intrinsic(width, height), width, height, self.depth_max
        )
        pcd_ = o3d.t.geometry.PointCloud(device)
        pcd_.point.positions = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(
            torch.utils.dlpack.from_dlpack(points.to_dlpack()).float()[mask_proj]
        )).to(device)
        frustum_block_coords = self.world.compute_unique_block_coordinates(
            pcd_, trunc_voxel_multiplier=8.0
        )
        buf_indices, _ = self.world.hashmap().find(frustum_block_coords)
        o3c.cuda.synchronize()
        return buf_indices

    def buf_coords(self, buf_indices):
        voxel_coords, _ = self.world.voxel_coordinates_and_flattened_indices(
            buf_indices
        )
        buf_coords = voxel_coords.reshape((-1, self.block_resolution**3, 3)).mean(1)
        return buf_coords

    def update(self, color, depth, extrinsic):
        color = o3d.t.geometry.Image(color).to(o3c.uint8).to(self.device)
        depth = o3d.t.geometry.Image(depth).to(o3c.uint16).to(self.device)
        extrinsic = o3c.Tensor.from_numpy(extrinsic)

        # Get active frustum block coordinates from input
        frustum_block_coords = self.world.compute_unique_block_coordinates(
            depth, self.intrinsic, extrinsic, self.depth_scale, self.depth_max
        )

        self.world.integrate(
            frustum_block_coords, depth, color, self.intrinsic,
            extrinsic, self.depth_scale, self.depth_max
        )
        return frustum_block_coords, extrinsic

    def take_snapshot(self, height, width, extrinsic, show=False, save_path=None, level="voxel"):
        assert level in ["block", "voxel", "pc"]
        img = torch.zeros(height, width, 3, dtype=torch.uint8)

        if level in ["block", "voxel"]:
            buf_indices = self.world.hashmap().active_buf_indices()
            voxel_coords, voxel_indices = self.world.voxel_coordinates_and_flattened_indices(
                buf_indices
            )
            o3c.cuda.synchronize()
            if level == "block":
                buf_coords = voxel_coords.reshape((-1, self.block_resolution**3, 3)).mean(1)
                v_proj, u_proj, _, mask_proj = self.get_points_in_fov(
                    buf_coords, extrinsic, self.custom_intrinsic(width, height), width, height, self.depth_max
                )
                color = self.world.attribute('color').reshape((-1, self.block_resolution**3, 3)).mean(1)
                indices = buf_indices.cpu().numpy()[mask_proj.cpu().numpy()]
            elif level == "voxel":
                v_proj, u_proj, _, mask_proj = self.get_points_in_fov(
                    voxel_coords, extrinsic, self.custom_intrinsic(width, height), width, height, self.depth_max
                )
                color = self.world.attribute('color').reshape((-1, 3))
                indices = voxel_indices.cpu().numpy()[mask_proj.cpu().numpy()]
            color = torch.utils.dlpack.from_dlpack(color.to_dlpack()).cpu()
            v_proj = v_proj.cpu()
            u_proj = u_proj.cpu()

            unique_indices, inverse_indices = torch.unique(v_proj * width + u_proj, return_inverse=True)
            sum_colors = torch.zeros_like(unique_indices, dtype=torch.float32).repeat(3,1).T
            sum_colors.index_add_(0, inverse_indices, color[indices])
            counts = torch.bincount(inverse_indices, minlength=len(unique_indices))
            avg_colors = sum_colors / counts.unsqueeze(1)
            img[(unique_indices // width), (unique_indices % width)] = avg_colors.to(torch.uint8)
        else:
            pcd = self.world.extract_point_cloud()
            points = pcd.point.positions
            color = pcd.point.colors * 255
            v_proj, u_proj, _, mask_proj = self.get_points_in_fov(
                points, extrinsic, self.custom_intrinsic(width, height), width, height, self.depth_max
            )
            v_proj, u_proj, indices = v_proj.cpu().numpy(), u_proj.cpu().numpy(), mask_proj.cpu().numpy()
            color = torch.utils.dlpack.from_dlpack(color.to_dlpack()).cpu()
            img[v_proj, u_proj] = color[indices].to(torch.uint8)

        if show or save_path is not None:
            ts.show([img], save=save_path is not None, file_path=save_path)
        return img

    @torch.no_grad()
    def fast_object_query(self, t_emb, points, colors=None, only_poi=False, topk=1, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def object_query(self, t_emb, points, colors=None, only_poi=False, topk=1, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def semantic_query(self, t_emb, points, colors=None, cmap=None):
        raise NotImplementedError

    @torch.no_grad()
    def instance_query(self, t_emb, points, colors=None, cmap=None):
        raise NotImplementedError

    @torch.no_grad()
    def panoptic_query(self, t_emb, points, colors=None, cmap=None):
        raise NotImplementedError


class VLState(BaseState):
    def __init__(
        self,
        intrinsic,
        depth_scale,
        depth_max,
        voxel_size = 5.0 / 512,
        block_resolution = 8,
        block_count = 100000,
        device = "CUDA:0",
        img_size = (640, 480),
        num_obj_points_per_block = 16, # increase if you have more memory
        matcher=None
    ) -> None:
        super().__init__(
            intrinsic, depth_scale, depth_max, voxel_size,
            block_resolution, block_count, device, img_size
        )
        # NOTE: number of sampled points in the block
        self.num_obj_points_per_block = num_obj_points_per_block
        self.emb_keys = torch.zeros(block_count, self.num_obj_points_per_block).long()
        self.emb_confs = torch.zeros(block_count, self.num_obj_points_per_block)
        self.emb_coords = torch.zeros(block_count, self.num_obj_points_per_block, 3)

        self.active_buf_indices = None
        self.emb_dict = torch.zeros(1, 512)
        self.emb_count = torch.ones(1)
        self.matcher = matcher

    def save(self, path):
        self.world.save(path)
        data = np.load(path)
        np.savez(
            path,
            intrinsic = self.intrinsic_np,
            extrinsic = np.array(self.poses),
            emb_keys = self.emb_keys.numpy(),
            emb_confs = self.emb_confs.numpy(),
            emb_coords = self.emb_coords.numpy(),
            emb_dict = self.emb_dict.numpy(),
            emb_count = self.emb_count.numpy(),
            **data
        )

    def load(self, path):
        self.world = self.world.load(path)
        data = np.load(path)
        self.intrinsic_np = data["intrinsic"]
        self.intrinsic = o3c.Tensor.from_numpy(self.intrinsic_np)
        self.poses = data["extrinsic"].tolist()

        self.emb_keys = torch.from_numpy(data["emb_keys"].reshape(-1, self.num_obj_points_per_block))
        self.emb_confs = torch.from_numpy(data["emb_confs"].reshape(-1, self.num_obj_points_per_block))
        self.emb_coords = torch.from_numpy(data["emb_coords"].reshape(-1, self.num_obj_points_per_block, 3))
        self.emb_dict = torch.from_numpy(data["emb_dict"])
        self.emb_count = torch.from_numpy(data["emb_count"])

    def adjust_embed_capacity(self):
        if self.world.hashmap().capacity() > self.emb_coords.shape[0]:
            delta = self.world.hashmap().capacity() - self.emb_coords.shape[0]
            self.emb_keys = torch.cat([self.emb_keys, torch.zeros(delta, self.num_obj_points_per_block).long()], dim=0)
            self.emb_confs = torch.cat([self.emb_confs, torch.zeros(delta, self.num_obj_points_per_block)], dim=0)
            self.emb_coords = torch.cat([self.emb_coords, torch.zeros(delta, self.num_obj_points_per_block, 3)], dim=0)

    @torch.no_grad()
    def update(self, color, depth, extrinsic, res_dict:dict={}):
        frustum_block_coords, extrinsic = super().update(color, depth, extrinsic)

        # NOTE: when the switch is off return without semantic integration
        if not res_dict:
            return

        self.adjust_embed_capacity()
        cur_buf_indices, _ = self.world.hashmap().find(frustum_block_coords) # (N,)
        o3c.cuda.synchronize()

        voxel_coords, _ = self.world.voxel_coordinates_and_flattened_indices(
            cur_buf_indices
        )
        cur_buf_indices = torch.utils.dlpack.from_dlpack(cur_buf_indices.to_dlpack())
        cur_buf_indices_cpu = cur_buf_indices.cpu()
        voxel_coords = torch.utils.dlpack.from_dlpack(voxel_coords.to_dlpack())

        cur_keys = self.emb_keys[cur_buf_indices_cpu] # (N, O)
        cur_confs = self.emb_confs[cur_buf_indices_cpu] # (N, O)
        cur_coords = self.emb_coords[cur_buf_indices_cpu] # (N, O, 3)
        unique_keys, inverse_unique_keys = torch.unique(cur_keys, return_inverse=True)

        # print(cur_keys.shape)
        # print(unique_keys)

        height, width = res_dict["conf_idx"].shape[:2]
        # NOTE: when there is nothing in the given scene
        if len(unique_keys) == 1:
            # NOTE: obtain xyz coords (N,3) from depth and valid mask of (H,W)
            obs_coords, mask = self.depth_to_point_cloud(
                depth, extrinsic, self.custom_intrinsic(width, height), width, height, self.depth_max, self.depth_scale
            )
            # NOTE: process model's outputs
            obs_keys = res_dict["conf_idx"][mask] # (N, )
            obs_conf = res_dict["conf_score"][mask] # (N, )

            # NOTE: remove background class
            non_zero_mask = obs_keys != 0
            obs_keys = obs_keys[non_zero_mask] # 1 ~
            obs_conf = obs_conf[non_zero_mask]
            obs_coords = obs_coords[non_zero_mask]

            # NOTE: find the corresponding buf indices (N',) for each xyz projected coords and valid mask of (N,)
            # NOTE: N' <= N as some of the projected coords may not be in the active blocks
            obs_buf_idx, valid = self.find_buf_indices_from_coord(
                cur_buf_indices,
                voxel_coords.view(-1, self.block_resolution**3, 3), # (M, 8^3, 3)
                obs_coords # (N, 3)
            )
            if len(obs_buf_idx) == 0:
                torch.cuda.empty_cache()
                print("[*] nothing can be integrated")
                return

             # NOTE: find the unique buf indices (unique blocks) and their corresponding counts (U,), (N',)
            unique_obs_buf, inverse_obs_ind = torch.unique(obs_buf_idx, return_inverse=True)
            sample_ind = self.random_sample_indices(inverse_obs_ind, self.num_obj_points_per_block)

            obs_keys = obs_keys[valid][sample_ind] # (L,)
            obs_conf = obs_conf[valid][sample_ind] # (L,)
            obs_coords = obs_coords[valid][sample_ind] # (L, 3)

            obs_buf_idx = obs_buf_idx[sample_ind] # (L,)
            inverse_obs_ind = inverse_obs_ind[sample_ind] # (L,)

            count = self.cumulative_count(inverse_obs_ind)
            obs_unique_, obs_keys_ = self.process_new_keys(obs_keys, start_id=len(self.emb_dict))

            self.emb_keys[obs_buf_idx, count] = obs_keys_.cpu()
            self.emb_confs[obs_buf_idx, count] = obs_conf.cpu()
            self.emb_coords[obs_buf_idx, count] = obs_coords.cpu()

            cap = res_dict["caption"][obs_unique_-1].cpu()
            self.emb_dict = torch.cat([self.emb_dict, cap], dim=0)
            self.emb_count = torch.cat([self.emb_count, torch.ones(len(cap))], dim=0)

            torch.cuda.empty_cache()
            return
        else:
            caps = res_dict["caption"]
            # NOTE: no object detected
            if len(caps) == 0:
                return
            # NOTE: obtain object-wise confidence images as res (H, W, X)
            obs_keys = res_dict["conf_idx"]
            obs_conf = res_dict["conf_score"]
            unique_obs_keys = torch.unique(obs_keys)
            res = torch.zeros(height, width, unique_obs_keys[-1]+1).to(obs_keys.device)
            res.scatter_(2, obs_keys.unsqueeze(-1), obs_conf.unsqueeze(-1))
            # NOTE: remove background class
            #! NOTE: index will be shifted by 1 (1 -> 0, 2 -> 1, ...)
            res = res[...,1:]

            if DBG:
                # print(res.shape)
                # print(unique_obs_keys)
                ts.show(res.permute(2,0,1), suptitle=f"res", save=True, file_path="res.png")

            # NOTE: obtain valid xyz coords (P, 3)
            object_mask = torch.nonzero(cur_keys.view(-1), as_tuple=True) # tuple(P, )
            emb_coords = cur_coords.view(-1, 3)[object_mask] # (P, 3)
            # NOTE: get points in FoV
            v_proj, u_proj, _, mask_proj = self.get_points_in_fov(
                o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(emb_coords)).to(self.device, o3c.float32),
                extrinsic, self.custom_intrinsic(width, height), width, height, self.depth_max
            )
            # NOTE: render the features in the Volume
            values = cur_confs.view(-1)[object_mask][mask_proj.cpu()]
            keys = inverse_unique_keys.view(-1)[object_mask][mask_proj.cpu()]
            img = torch.zeros(height, width, len(unique_keys)).to(obs_conf.device)
            img[
                v_proj,
                u_proj,
                keys
            ] = values.to(obs_conf.device)
            # NOTE: filter blank images
            blank_filter = (img != 0).sum(0).sum(0) != 0
            img = img[...,blank_filter]
            unique_keys = unique_keys[blank_filter.cpu()]
            embs = self.emb_dict[unique_keys].to(obs_conf.device)

            if DBG:
                ts.show(img.permute(2,0,1), suptitle=f"rend img", save=True, file_path="img.png")

            # NOTE: match the current predictions with the rendered features
            matches = self.matcher(caps, res.permute(2,0,1), embs, img.permute(2,0,1))
            #! NOTE: shift back the obs index by 1 (0 -> 1, 1 -> 2, ...)
            res_match_key = matches[0] + 1
            img_match_key = unique_keys[matches[1]]

            if DBG:
                print("matched: ", res_match_key)
                ts.show(res.permute(2,0,1)[res_match_key-1], suptitle=f"matched")
            if DBG:
                ts.show(img.permute(2,0,1)[matches[1]], suptitle=f"matched-pair")

            # NOTE: obtain xyz coords (N,3) from depth and valid mask of (H,W)
            obs_coords, mask = self.depth_to_point_cloud(
                depth, extrinsic, self.custom_intrinsic(width, height),
                width, height, self.depth_max, self.depth_scale
            )
            # NOTE: process model's outputs
            obs_keys = obs_keys[mask] # (N, )
            obs_conf = obs_conf[mask] # (N, )

            res_keys = torch.zeros_like(obs_keys, device=obs_keys.device)
            # NOTE: when unmatched, register obs_keys as new keys
            key_offset = len(self.emb_dict)
            unmatched_keys = [k for k in reversed(range(1,res.shape[-1]+1)) if k not in res_match_key]

            if DBG:
                print("unmatched: ", unmatched_keys)
                ts.show(res.permute(2,0,1)[torch.tensor(unmatched_keys)-1], suptitle=f"unmatched")

            # NOTE: when unmatched, register obs_keys as new keys
            if unmatched_keys:
                for i, k in enumerate(unmatched_keys):
                    if DBG:
                        print(f"{k} -> {key_offset + i}")
                        ts.show([res.permute(2,0,1)[k-1]], suptitle=f"{k} -> {key_offset + i}")
                    res_keys[obs_keys == k] = key_offset + i
                # NOTE: add new embs to emb dict
                self.emb_dict = torch.cat([self.emb_dict, caps[torch.tensor(unmatched_keys)-1].cpu()], dim=0)
                self.emb_count = torch.cat([self.emb_count, torch.ones(len(unmatched_keys))], dim=0)

            # NOTE: when matched, overwrite the obs_keys with the corresponding cur_keys
            for r_key, i_key in zip(res_match_key, img_match_key):
                res_keys[obs_keys == r_key] = i_key

            if DBG:
                print("updated: ", torch.unique(res_keys))

            # NOTE: remove background class
            non_zero_mask = res_keys != 0
            res_keys = res_keys[non_zero_mask]
            obs_conf = obs_conf[non_zero_mask]
            obs_coords = obs_coords[non_zero_mask]

            # NOTE: find the corresponding buf indices (N',) for each xyz projected coords and valid mask of (N,)
            # NOTE: combine the existing records with the new observations
            comb_coords = torch.cat([
                obs_coords,
                emb_coords[mask_proj.cpu()].to(obs_coords.device)
            ], dim=0)

            if DBG:
                print(voxel_coords.view(-1, self.block_resolution**3, 3).shape, comb_coords.shape, obs_coords.shape, emb_coords.shape)
            comb_buf_idx, valid = self.find_buf_indices_from_coord(
                cur_buf_indices,
                voxel_coords.view(-1, self.block_resolution**3, 3), # (M, 8^3, 3)
                comb_coords # (N, 3)
            )
            if len(comb_buf_idx) == 0:
                torch.cuda.empty_cache()
                print("[*] nothing can be integrated")
                return

            unique_comb_buf, inverse_comb_ind = torch.unique(comb_buf_idx, return_inverse=True)
            sample_ind = self.random_sample_indices(inverse_comb_ind, self.num_obj_points_per_block)

            comb_keys = torch.cat([
                res_keys,
                cur_keys.view(-1)[object_mask][mask_proj.cpu()].to(res_keys.device)
            ], dim=0)[valid][sample_ind].cpu() # (L,)
            comb_conf = torch.cat([
                obs_conf,
                values.to(obs_conf.device)
            ], dim=0)[valid][sample_ind].cpu() # (L,)
            comb_coords = comb_coords[valid][sample_ind].cpu() # (L, 3)
            comb_buf_idx = comb_buf_idx[sample_ind].cpu() # (L,)
            inverse_comb_ind = inverse_comb_ind[sample_ind] # (L,)

            count = self.cumulative_count(inverse_comb_ind).cpu()

            self.emb_keys[comb_buf_idx, count] = comb_keys
            self.emb_confs[comb_buf_idx, count] = comb_conf
            self.emb_coords[comb_buf_idx, count] = comb_coords

            torch.cuda.empty_cache()
            return

    def process_new_keys(self, keys, start_id=1):
        """ process keys to ensure that they are continuous before adding to dict
        Args:
            keys (torch.Tensor): tensor of keys
            start_id (int, optional): starting id. Defaults to 1.
        """
        unique_keys, inverse = torch.unique(keys, return_inverse=True)
        return unique_keys, torch.arange(start_id, start_id+len(unique_keys), device=keys.device)[inverse]

    @staticmethod
    def random_sample_indices(input_tensor, n):
        unique_keys = torch.unique(input_tensor)
        sampled_indices = []

        for key in unique_keys:
            key_indices = (input_tensor == key).nonzero(as_tuple=True)[0]
            if len(key_indices) > n:
                key_indices = key_indices[torch.randperm(len(key_indices))[:n]]
            sampled_indices.append(key_indices)

        sampled_indices = torch.cat(sampled_indices)
        return sampled_indices

    def find_buf_indices_from_coord(self, buf_indices, voxel_coords, coordinates):
        """
        Finds the index of the cube that incorporates each coordinate in a batched manner.

        Args:
            buf_indices (torch.Tensor): N tensor of buf indices for the given voxel coords
            voxel_coords (torch.Tensor): Nx8^3x3 tensor where N is the number of cubes.
            coordinates (torch.Tensor): Mx3 tensor where M is the number of coordinates. (usually M >> N)

        Returns:
            tensor: M tensor that contains the index of the cube that incorporates each coordinate.
        """
        # NOTE: find min and max of x, y, z for each cube
        # min_vals = torch.min(voxel_coords, dim=1).values  # Shape: Nx3
        # max_vals = torch.max(voxel_coords, dim=1).values  # Shape: Nx3
        # NOTE: account for border coordinates
        min_vals = voxel_coords[:, 0, :]  - self.voxel_size/2 # Shape: Nx3
        max_vals = voxel_coords[:, -1, :] + self.voxel_size/2 # Shape: Nx3

        # NOTE: check if each coordinate is inside each cube
        is_inside = (min_vals[:, None] <= coordinates[None]) & (coordinates[None] < max_vals[:, None]) # Shape: NxMx3
        # NOTE: all coordinates must be inside the cube along all 3 dimensions (x, y, z)
        is_inside_all_dims = torch.all(is_inside, dim=2).long()  # Shape: NxM
        # NOTE: find cube index for each coordinate
        cube_idx_for_each_coord = buf_indices[torch.argmax(is_inside_all_dims, dim=0)]  # Shape: M
        # NOTE: find valid mask where a cube was found for a coordinate
        valid_mask = torch.any(is_inside_all_dims, dim=0)  # Shape: M
        return cube_idx_for_each_coord[valid_mask], valid_mask

    @staticmethod
    def cumulative_count(tensor):
        """cumulative count of unique elements in a tensor
        Example:
        input > tensor([0, 2, 1, 2, 2, 3])
        output > tensor([0, 0, 0, 1, 2, 0])
        Args:
            tensor (torch.Tensor): 1D tensor usually inverse indices
        """
        unique_elements, counts = torch.unique(tensor, return_counts=True)
        output = torch.zeros_like(tensor)
        for unique, count in zip(unique_elements.tolist(), counts.tolist()):
            indices = (tensor == unique).nonzero(as_tuple=True)[0]
            # NOTE: 0-based index counting
            output[indices] = torch.arange(count, device=tensor.device)
        return output

    @torch.no_grad()
    def fast_object_query(self, t_emb, points, colors=None, only_poi=False, topk=1):
        """
        Args:
            t_emb (torch.Tensor): query embedding
            points (np.array): xyz array
            colors (np.array): color array
            only_poi (bool, optional): return only points of interest. Defaults to False.
            topk (int, optional): take topk similar regions as predication. Defaults to 1.
        """
        buf_indices = self.world.hashmap().active_buf_indices()
        buf_indices = torch.utils.dlpack.from_dlpack(buf_indices.to_dlpack())

        mask_key = self.emb_keys[
            buf_indices.cpu()
        ].view(-1, self.num_obj_points_per_block)

        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        valid_keys = torch.unique(mask_key)
        embed = self.emb_dict[valid_keys].to(t_emb.device)
        mask_pred_caption = embed / (embed.norm(dim=-1, keepdim=True) + 1e-7)
        mask_cls = torch.einsum("cd,nd->cn", t_emb, mask_pred_caption)
        value, indices = torch.topk(mask_cls, topk, dim=1)

        key_loc = torch.zeros_like(mask_key).bool()
        for idx in valid_keys[indices.flatten(0).cpu()]:
            key_loc |= mask_key == idx

        poi = self.emb_coords[buf_indices.cpu()].view(
            -1, self.num_obj_points_per_block, 3
        )[key_loc.unsqueeze(-1).repeat(1,1,3)].view(-1,3).numpy()
        if only_poi:
            return poi

        points = np.concatenate([points, poi], axis=0)
        colors = np.concatenate([colors, np.ones_like(poi) * np.array([[255, 0, 0]])], axis=0)
        return points, colors

    @torch.no_grad()
    def object_query(
        self, t_emb, points, colors=None, only_poi=False, topk=1, dist_thresh=0.1, count_thresh=3
    ):
        poi = self.fast_object_query(t_emb, points, colors, only_poi=True, topk=topk)
        tree = KDTree(poi, leaf_size=10)
        dist, ind = tree.query(points, k=10)

        mask = np.sum(dist <= dist_thresh, axis=1) > count_thresh
        if only_poi:
            return points[mask]

        colors[mask] = np.array([255, 0, 0])
        return points, colors

    @torch.no_grad()
    def semantic_query(self, t_emb, points, colors=None, cmap=None):
        buf_indices = self.world.hashmap().active_buf_indices()
        buf_indices = torch.utils.dlpack.from_dlpack(buf_indices.to_dlpack()).cpu()
        torch.cuda.empty_cache()

        mask_key = self.emb_keys[
            buf_indices
        ].view(-1, self.num_obj_points_per_block) # (N, O)
        mask_conf = self.emb_confs[
            buf_indices
        ].view(-1, self.num_obj_points_per_block) # (N, O)

        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7) # (C, D)
        # NOTE: exclude BG in semantic mode
        valid_keys = torch.unique(mask_key)[1:]
        embed = self.emb_dict[valid_keys].to(t_emb.device)
        mask_pred_caption = embed / (embed.norm(dim=-1, keepdim=True) + 1e-7) # (E, D)
        mask_cls = torch.einsum("cd,nd->cn", t_emb, mask_pred_caption) # (C, E)
        outputs_class = F.softmax(mask_cls, dim=0) # (C, E)

        mask_key = mask_key.flatten(0)
        mask_pred = torch.zeros(mask_key.shape[0], torch.unique(mask_key)[-1]+1, device=t_emb.device)
        mask_pred.scatter_(1, mask_key.to(t_emb.device).unsqueeze(-1), mask_conf.view(-1,1).to(t_emb.device))
        semseg = torch.einsum("cn,qn->qc", outputs_class, mask_pred[:, valid_keys]).argmax(1).cpu() # (M,)

        poi = self.emb_coords[
            buf_indices
        ].view(-1, 3)
        poi = poi[mask_key != 0].view(-1,3).numpy() # (M',)
        semseg = semseg[mask_key != 0].numpy()

        tree = KDTree(poi, leaf_size=10)
        dist, ind = tree.query(points, k=10)

        cls = [cmap(np.argmax(np.bincount(m, weights=1/(1+d)))) for d, m in zip(dist, semseg[ind])]
        colors = np.array(cls)[:,:3]
        return points, colors

    @torch.no_grad()
    def panoptic_query(self, t_emb, points, colors=None, cmap=None, metadata=None, overlap_threshold=0.7):
        buf_indices = self.world.hashmap().active_buf_indices()
        buf_indices = torch.utils.dlpack.from_dlpack(buf_indices.to_dlpack()).cpu()
        torch.cuda.empty_cache()

        mask_key = self.emb_keys[
            buf_indices
        ].view(-1, self.num_obj_points_per_block) # (N, O)
        mask_conf = self.emb_confs[
            buf_indices
        ].view(-1, self.num_obj_points_per_block) # (N, O)

        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7) # (C, D)
        # NOTE: exclude BG in semantic mode
        valid_keys = torch.unique(mask_key)[1:]
        embed = self.emb_dict[valid_keys].to(t_emb.device)
        mask_pred_caption = embed / (embed.norm(dim=-1, keepdim=True) + 1e-7) # (E, D)
        mask_cls = torch.einsum("cd,nd->cn", t_emb, mask_pred_caption) # (C, E)
        scores, labels = F.softmax(mask_cls, dim=0).max(0) # (E,)
        # NOTE: object mask threshold
        keep = scores > 0.0
        cur_scores = scores[keep] # (E',)
        cur_classes = labels[keep] # (E',)

        mask_key = mask_key.flatten(0)
        mask_pred = torch.zeros(mask_key.shape[0], torch.unique(mask_key)[-1]+1, device=t_emb.device)
        mask_pred.scatter_(1, mask_key.to(t_emb.device).unsqueeze(-1), mask_conf.view(-1,1).to(t_emb.device))
        mask_pred = mask_pred[:, valid_keys] # (M, E)
        cur_masks = mask_pred[:, keep] # (M, E')

        cur_prob_masks = cur_scores.view(1, -1) * cur_masks # (M, E')

        current_segment_id = 0
        panoptic_seg = torch.zeros(cur_masks.shape[0], device=cur_masks.device).long()
        cur_mask_ids = cur_prob_masks.argmax(-1) # (M, )
        stuff_memory_list = {}
        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[:, k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)

            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < overlap_threshold:
                    print(f"low overlap {mask_area / original_area} below {overlap_threshold}")
                    continue

                if metadata is not None:
                    isthing = pred_class in metadata.thing_dataset_id_to_contiguous_id.values()
                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id

        poi = self.emb_coords[buf_indices].view(-1, 3)
        poi = poi[mask_key != 0].view(-1,3).numpy() # (M',)
        panoptic_seg = panoptic_seg[mask_key != 0].cpu().numpy()

        tree = KDTree(poi, leaf_size=10)
        dist, ind = tree.query(points, k=15)

        cmap = rand_cmap(current_segment_id, type="bright", first_color_black=False)
        cls = [cmap(np.argmax(np.bincount(m, weights=1/(1+d)))) for d, m in zip(dist, panoptic_seg[ind])]
        colors = np.array(cls)[:,:3]
        return points, colors


class CFState(BaseState):
    """ Point (Block) wise embedding fusion
    """
    def __init__(
        self,
        intrinsic,
        depth_scale,
        depth_max,
        voxel_size = 5.0 / 512,
        block_resolution = 8,
        block_count = 100000,
        dim = 512,
        device = "CUDA:0",
        img_size=(640, 480)
    ) -> None:
        super().__init__(
            intrinsic, depth_scale, depth_max, voxel_size,
            block_resolution, block_count, device, img_size
        )
        # store embeddings on each block to avoid GPU memory overflow
        self.embed = torch.zeros(block_count, dim)
        self.weight = torch.zeros(block_count, 1)
        self.dim = dim

    def adjust_embed_capacity(self):
        if self.world.hashmap().capacity() > self.embed.shape[0]:
            delta = self.world.hashmap().capacity() - self.embed.shape[0]
            self.embed = torch.cat([self.embed, torch.zeros(delta, self.dim)], dim=0)
            self.weight = torch.cat([self.weight, torch.zeros(delta, 1)], dim=0)

    def update(self, color, depth, extrinsic, res_dict:dict={}):
        frustum_block_coords, extrinsic = super().update(color, depth, extrinsic)

        # NOTE: when the switch is off return without semantic integration
        if not res_dict:
            return

        self.adjust_embed_capacity()
        cur_buf_indices, _ = self.world.hashmap().find(frustum_block_coords)
        o3d.core.cuda.synchronize()
        voxel_coords, _ = self.world.voxel_coordinates_and_flattened_indices(
            cur_buf_indices
        )
        voxel_coords = torch.utils.dlpack.from_dlpack(voxel_coords.to_dlpack())

        cur_buf_coords = self.buf_coords(cur_buf_indices)
        # sample coords with virtual camera
        height, width = res_dict["emb"].shape[:2]
        v_proj, u_proj, _, mask_proj = self.get_points_in_fov(
            cur_buf_coords, extrinsic, self.custom_intrinsic(width, height), width, height, self.depth_max
        )

        cur_buf_indices, v_proj, u_proj = \
            cur_buf_indices.cpu().numpy()[mask_proj.cpu().numpy()], v_proj.cpu(), u_proj.cpu()

        with torch.no_grad():
            w = self.weight[cur_buf_indices] * 0.9
            wp = w + 1

            self.embed[cur_buf_indices]  = (
                self.embed[cur_buf_indices] * w +
                res_dict["emb"][v_proj, u_proj].cpu()
            ) / (wp)

            self.weight[cur_buf_indices] = wp

    @torch.no_grad()
    def fast_object_query(
        self, t_emb, points, colors, only_poi=False, obj_thresh=0.5, **kwargs
    ):
        """ obtain heatmap of relevence between map and query
        Args:
            t_emb (torch.Tensor): text embedding
        """
        assert t_emb.shape[0] == 1, "[*] only support single query"
        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        buf_indices = self.active_buf_indices()
        buf_coords = self.buf_coords(buf_indices)
        buf_indices = torch.utils.dlpack.from_dlpack(
            buf_indices.to_dlpack()
        )
        buf_coords = torch.utils.dlpack.from_dlpack(
            buf_coords.to_dlpack()
        )
        embed = self.embed.to(t_emb.device)[buf_indices]
        mask_pred_caption = embed / (embed.norm(dim=-1, keepdim=True) + 1e-7)
        out = torch.einsum("cd,nd->cn", t_emb, mask_pred_caption).flatten().cpu() # (N,)

        if only_poi:
            return buf_coords[out > obj_thresh].cpu().numpy()

        out = out.cpu().numpy()
        out = (out - np.min(out)) / np.max(out)
        cmap = plt.get_cmap("plasma")
        colors = np.array([(np.array(cmap(v)[:3])) for v in out])
        return buf_coords.cpu().numpy(), colors

    @torch.no_grad()
    def object_query(
        self, t_emb, points, colors=None, cmap=None, obj_thresh=0.1, **kwargs
    ):
        if cmap is None:
            cmap = plt.get_cmap("plasma")
        buf_coords, colors = self.fast_object_query(
            t_emb, points, colors, only_poi=False, obj_thresh=obj_thresh
        )

        tree = KDTree(buf_coords.reshape(-1,3), leaf_size=10)
        dist, ind = tree.query(points, k=5)

        colors = np.array([np.mean(m, axis=0) for m in colors[ind]])
        return points, colors

    @torch.no_grad()
    def semantic_query(self, t_emb, points, colors, cmap=None):
        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        buf_indices = self.active_buf_indices()
        buf_coords = self.buf_coords(buf_indices)
        buf_indices = torch.utils.dlpack.from_dlpack(
            buf_indices.to_dlpack()
        )
        embed = self.embed.to(t_emb.device)[buf_indices]
        mask_pred_caption = embed / (embed.norm(dim=-1, keepdim=True) + 1e-7)
        semseg = torch.einsum("cd,nd->cn", t_emb, mask_pred_caption).argmax(0).cpu().numpy() # (N,)

        tree = KDTree(buf_coords.cpu().numpy().reshape(-1,3), leaf_size=10)
        dist, ind = tree.query(points, k=3)

        cls = [cmap(np.argmax(np.bincount(m, weights=1/(1+d)))) for d, m in zip(dist, semseg[ind])]
        colors = np.array(cls)[:,:3]
        return points, colors