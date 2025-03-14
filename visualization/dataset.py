
import os
import numpy as np
from pathlib import Path
from torch.utils import data
import yaml
import pickle
from mmcv.image.io import imread
from copy import deepcopy
from typing import Dict, List, Tuple, Any
import random

import torch
import numba as nb
from torch.utils import data
import numpy as np
from scipy.spatial.transform import Rotation
from alive_progress import alive_bar
from dataloader.transform_3d import PadMultiViewImage, \
    NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage
from dataloader.flink_dataset_loader import FlinkDatasetLoader, FlinkDatapoint, MetadataDetails

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]

test_pipeline = [
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]


class ImagePoint_NuScenes_vis(data.Dataset):
    def __init__(self, data_path, imageset='train', 
                 scene_idx=None, scene_name=None,
                 label_mapping="nuscenes.yaml", nusc=None):
        self.return_ref = False

        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        with open(label_mapping, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        nusc_infos = data['infos']

        # insert sweep frames between keyframes
        if scene_idx is not None or scene_name is not None:
            scene_name = list(nusc_infos.keys())[scene_idx] if scene_name is None else scene_name
            print(f'visualizing {scene_name}')
            self.nusc_infos = nusc_infos[scene_name]
            nusc_infos = deepcopy(self.nusc_infos)

            sweep_cams = []
            sweep_tss = []
            reverse_tab = {
                'CAM_FRONT':0, 
                'CAM_FRONT_RIGHT':1, 
                'CAM_FRONT_LEFT':2, 
                'CAM_BACK':3, 
                'CAM_BACK_LEFT':4, 
                'CAM_BACK_RIGHT':5
            }
            for cam_type in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
                dir = os.path.join(data_path, 'sweeps', cam_type)
                filenames = os.listdir(dir)
                files = [os.path.join(dir, fn) for fn in filenames]
                ts = [int(fn.split('__')[-1].split('.')[0]) for fn in filenames]
                idx = np.argsort(ts)
                sweep_cams.append(np.array(files)[idx])
                sweep_tss.append(np.array(ts)[idx])
            sweep_cams = np.array(sweep_cams)
            sweep_tss = np.array(sweep_tss)

            for i in range(len(self.nusc_infos)-1):
                insert_items = []
                start_ts = self.nusc_infos[i]['timestamp']
                end_ts = self.nusc_infos[i+1]['timestamp']
                temp_cams = []
                for sweep_cam, sweep_ts in zip(sweep_cams, sweep_tss):
                    temp_cam = sweep_cam[[(ts < end_ts and ts > start_ts) for ts in sweep_ts]]
                    temp_cams.append(temp_cam.tolist())
                min_len = min([len(temp_cam) for temp_cam in temp_cams])
                temp_cams = [temp_cam[:min_len] for temp_cam in temp_cams]
                for j in range(min_len):
                    temp_dict = deepcopy(self.nusc_infos[i])
                    for cam_type, cam_info in temp_dict['cams'].items():
                        cam_info['data_path'] = temp_cams[reverse_tab[cam_type]][j]
                    temp_dict['timestamp'] = temp_cams[0][j].split('__')[-1].split('.')[0]
                    insert_items.append(temp_dict)
                nusc_infos.extend(insert_items)
        
        self.nusc_infos = nusc_infos
        
        self.data_path = data_path
        self.lidarseg_path = data_path
        self.nusc = nusc
        self.cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        imgs_info = self.get_data_info(info)
        img_metas = {
            'lidar2img': imgs_info['lidar2img'],
            'cam_positions': imgs_info['cam_positions'],
            'focal_positions': imgs_info['focal_positions']
        }
        # read 6 cams
        imgs = []
        for filename in imgs_info['img_filename']:
            imgs.append(
                imread(filename, 'unchanged').astype(np.float32)
            )
        
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidarseg_labels_filename = os.path.join(self.lidarseg_path, self.nusc.get('lidarseg', lidar_sd_token)['filename'])
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        
        lidar_path = info['lidar_path']
        points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])

        data_tuple = (imgs, img_metas, points[:, :3], points_label.astype(np.uint8))

        # deal with scene
        scene_token = self.nusc.get('sample', info['token'])['scene_token']
        scene_meta = self.nusc.get('scene', scene_token)
        timestamp = info['timestamp']
        return data_tuple, imgs_info['img_filename'], scene_meta, timestamp
    

    def get_data_info(self, info):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        # standard protocal modified from SECOND.Pytorch
        f = 0.0055
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
        )

        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam_positions = []
        focal_positions = []
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info[
                'sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)

            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)

            cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
            cam_positions.append(cam_position.flatten()[:3])
            focal_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
            focal_positions.append(focal_position.flatten()[:3])

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                cam_positions=cam_positions, # w, h, z, meters,
                focal_positions=focal_positions
            ))

        return input_dict
    
class ImagePoint_FLINK_vis(data.Dataset):
    CATEGORY_STR_TO_ID = {
        "box": 2,
        "environment": 1,
    }
    """
    Dataset class for loading Flink data.

    Args:
        data_path (str): Path to the root directory of the Flink dataset.
        label_mapping (str, optional):  Not used in this implementation, kept for consistency.
        len_dataset (int, optional): Length of the dataset. Defaults to None.
        img_num (int, optional): Number of images to sample. Defaults to 6.
        device (torch.device, optional): Device to load the data on. Defaults to 'cuda'.
    """
    def __init__(self, data_path: str, label_mapping: str = "nuscenes.yaml", len_dataset: int | None = None, img_num: int = 6):
        self.data_path: Path = Path(data_path)
        self.label_mapping: str = label_mapping  # Not used, but kept for API consistency
        self.img_num: int = img_num
        self.dataset_loaders: list[tuple[str, FlinkDatasetLoader]] = []

        REQUIRED_FOLDERS = {'depth', 'images', 'labels', 'metadata'}

        valid_dataset_paths: List[Path] = []
        def check_directory(dir_path: Path):
            # Check if this directory contains any of the required folders
            if any((dir_path / folder).exists() for folder in REQUIRED_FOLDERS):
                valid_dataset_paths.append(dir_path)
                self.dataset_loaders.append((dir_path, None))
                return
            # Recursively check subdirectories
            for item in dir_path.iterdir():
                if item.is_dir():
                    check_directory(item)
        check_directory(self.data_path)
        with alive_bar(len(valid_dataset_paths), title="Loading dataset") as bar:
            for dataset_path in valid_dataset_paths:
                self.dataset_loaders[dataset_path] = FlinkDatasetLoader(dataset_path)
                bar()
        
        # self.len_dataset = 10000
        if len_dataset is not None:
            self.len_dataset = len_dataset
        else:
            self.len_dataset = sum(len(loader) for loader in self.dataset_loaders.values())

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.len_dataset
    
    def get_datapoint(self, dataset_idx:int, camera_idxs:List[int]) -> Tuple[List[np.ndarray], Dict[str, Any], np.ndarray, np.ndarray]:
        _, selected_dataset = self.dataset_loaders[dataset_idx]
        selected_datapoints: List[FlinkDatapoint] = [selected_dataset[i] for i in camera_idxs]

        def metadata_to_posematrix(metadata: MetadataDetails) -> np.ndarray:
            rotation: np.ndarray = np.array(metadata.rotation)
            position: np.ndarray = np.array(metadata.position)
            # Create 4x4 transformation matrix
            # Convert exponential coordinates (axis-angle) to rotation matrix
            R = Rotation.from_rotvec(rotation).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = position
            return T

        imgs: List[np.ndarray] = []
        lidar2imgs: List[np.ndarray] = []
        cam_positions: List[np.ndarray] = []
        focal_positions: List[np.ndarray] = []
        
        all_points: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        for datapoint in selected_datapoints:
            # assume world center is the lidar
            cam2lidar: np.ndarray = metadata_to_posematrix(datapoint.metadata.metadata)
            lidar2cam: np.ndarray = np.linalg.inv(cam2lidar)
            viewpad = np.eye(4)
            viewpad[:3, :3] = datapoint.metadata.metadata.camera_matrix
            lidar2img = (viewpad @ lidar2cam)
            lidar2imgs.append(lidar2img)
            cam_positions.append((cam2lidar @ np.array([0., 0., 0., 1.]).reshape([4, 1])).flatten()[:3])
            f = viewpad[0, 0]
            focal_positions.append((lidar2cam @ np.array([0., 0., f, 1.]).reshape([4, 1])).flatten()[:3])
            imgs.append(datapoint.get_image().astype(np.float32))
            
            depth = datapoint.get_depth()
            
            points, valid_points = self._depth_to_pointcloud(depth, np.array(datapoint.metadata.metadata.camera_matrix), cam2lidar)
            labels = np.ones((depth.shape[0], depth.shape[1]), dtype=np.uint8)
            for segment in datapoint.label_data.segmentations:
                bbox = segment.bbox
                mask = ImagePoint_FLINK_vis._rle_to_mask(segment.mask, (bbox[2], bbox[3]))
                if bbox is not None and mask is not None:
                    # Extract bbox coordinates
                    x, y, w, h = bbox
                    # Create mask array within bbox
                    mask_array = np.array(mask).reshape(h, w)
                    # Fill the bbox region with mask values
                    labels[y:y+h, x:x+w][mask_array == 1] = self.CATEGORY_STR_TO_ID[segment.category_id]
            labels = labels.reshape(-1, 1)

            points = points[valid_points]
            labels = labels[valid_points]
            all_points.append(points)
            all_labels.append(labels)

        combined_points: np.ndarray = np.concatenate(all_points, axis=0)
        combined_labels: np.ndarray = np.concatenate(all_labels, axis=0)
            
        # Randomly sample points if we have more than that
        if len(combined_points) > 200000:
            sample_indices = np.random.choice(len(combined_points), 200000, replace=False)
            combined_points = combined_points[sample_indices]
            combined_labels = combined_labels[sample_indices]

        img_metas: Dict[str, Any] = {
            'lidar2img': lidar2imgs,
            'cam_positions': cam_positions,
            'focal_positions': focal_positions,
            'raw_points': combined_points,
            'raw_labels': combined_labels
        }  # Placeholder for consistency

        data_tuple: Tuple[List[np.ndarray], Dict[str, List[np.ndarray]], np.ndarray, np.ndarray] = (imgs, img_metas, combined_points, combined_labels)
        return data_tuple, ["fake_filename"], "fake_scene_meta", None

    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], Dict[str, Any], np.ndarray, np.ndarray]:
        """
        Get a sample from the dataset.

        Args:

        Returns:
            Tuple[List[np.ndarray], Dict[str, Any], np.ndarray, np.ndarray]: A tuple containing:
                - List[np.ndarray]: List of 6 images (as numpy arrays).
                - Dict[str, Any]:  Empty dictionary (for consistency with NuScenes).
                - np.ndarray:  The combined point cloud data (x, y, z coordinates).
                - np.ndarray:  The point cloud labels (all ones).
        """
        # sample a dataset from the dataset_loaders
        selected_dataset_idx = random.randint(0, len(self.dataset_loaders)-1)
        selected_dataset_path, selected_dataset = self.dataset_loaders[selected_dataset_idx]
        if selected_dataset is None:
            selected_dataset = FlinkDatasetLoader(selected_dataset_path)
            self.dataset_loaders[selected_dataset_idx] = (selected_dataset_path, selected_dataset)
        # sample img_num images from the selected dataset
        datapoint_indices: List[int] = random.sample(range(len(selected_dataset)), self.img_num)
        return self.get_datapoint(selected_dataset_idx, datapoint_indices)
    
    @staticmethod
    def _rle_to_mask(rle: List[int], shape: Tuple[int, int]) -> np.ndarray:
        """Convert RLE to binary mask."""
        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        current = 0
        for i in range(0, len(rle), 2):
            length = rle[i]
            data = rle[i + 1]
            start = current
            mask[start:start + length] = data
            current = start + length
        return mask.reshape(shape)

    def _depth_to_pointcloud(self, depth_image: np.ndarray, camera_matrix: np.ndarray, world2cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a depth image to a point cloud in world coordinates.

        Args:
            depth_image (np.ndarray): Depth image (in millimeters).
            camera_matrix (np.ndarray): 3x3 camera intrinsic matrix.
            world2cam (np.ndarray): 4x4 transformation matrix from world to camera coordinates.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Point cloud data (Nx3 array of x, y, z coordinates).
                - Valid points mask (N,) array.
        """
        height, width = depth_image.shape[:2]
        fx: float = camera_matrix[0, 0]
        fy: float = camera_matrix[1, 1]
        cx: float = camera_matrix[0, 2]
        cy: float = camera_matrix[1, 2]

        # Create meshgrid of pixel coordinates
        v, u = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert depth to meters
        depth = (depth_image / 1000.0)
        
        # Calculate x,y,z coordinates
        x = (v - cx) * depth / fx
        y = (u - cy) * depth / fy
        z = depth
        
        # Stack coordinates
        points = np.stack([x, y, z], axis=-1)
        
        # Reshape to (N,3)
        points = points.reshape(-1, 3)
        
        # Filter out invalid points
        valid_points = depth.reshape(-1) > 1e-5
        
        # Transform to world coordinates
        points = (world2cam[:3, :3] @ points.T).T + world2cam[:3, 3]
        
        return points.astype(np.float32), valid_points

class DatasetWrapper_NuScenes_vis(data.Dataset):
    def __init__(self, in_dataset, grid_size, ignore_label=0, fixed_volume_space=False, 
                 max_volume_space=[50, np.pi, 3], min_volume_space=[0, -np.pi, -5], phase='train'):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size).astype(np.int32)
        self.ignore_label = ignore_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.polar = False

        if phase == 'train':
            transforms = [
                PhotoMetricDistortionMultiViewImage(),
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=32)
            ]
        else:
            transforms = [
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=32)
            ]
        self.transforms = transforms

    def __len__(self):
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        data, filelist, scene_meta, timestamp = self.point_cloud_dataset[index]
        imgs, img_metas, xyz, labels = data
        
        # deal with img augmentations
        imgs_dict = {'img': imgs}
        for t in self.transforms:
            imgs_dict = t(imgs_dict)
        imgs = imgs_dict['img']
        imgs = [img.transpose(2, 0, 1) for img in imgs]
        img_metas['img_shape'] = imgs_dict['img_shape']

        xyz_pol = xyz
        
        assert self.fixed_volume_space
        max_bound = np.asarray(self.max_volume_space)  # 51.2 51.2 3
        min_bound = np.asarray(self.min_volume_space)  # -51.2 -51.2 -5
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size                 # 200, 200, 16
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        # TODO: grid_ind of float dtype may be better.
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (imgs, img_metas, processed_label, grid_ind, labels)

        return data_tuple, filelist, scene_meta, timestamp


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label
