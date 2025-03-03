import os
import yaml
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

from mmcv.image.io import imread
from torch.utils import data
import numpy as np
from scipy.spatial.transform import Rotation
from alive_progress import alive_bar

from dataloader.flink_dataset_loader import FlinkDatasetLoader, FlinkDatapoint, MetadataDetails

class ImagePoint_NuScenes(data.Dataset):
    def __init__(self, data_path, imageset='train', label_mapping="nuscenes.yaml", nusc=None):
        with open(imageset, 'rb') as f:
            data_raw = pickle.load(f)

        data = {}
        data["metadata"] = data_raw["metadata"]
        data["infos"] = []
        for info in data_raw["infos"]:
            if os.path.exists(info["lidar_path"]):
                data["infos"].append(info)

        with open(label_mapping, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        self.nusc_infos = data['infos']
        self.data_path = data_path
        self.nusc = nusc

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        imgs_info = self.get_data_info(info)
        img_metas = {
            'lidar2img': imgs_info['lidar2img'],
        }
        # read 6 cams
        imgs = []
        for filename in imgs_info['img_filename']:
            imgs.append(
                imread(filename, 'unchanged').astype(np.float32)
            )

        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidarseg_labels_filename = os.path.join(self.data_path, self.nusc.get('lidarseg', lidar_sd_token)['filename'])
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        
        lidar_path = info['lidar_path']        
        points = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])

        data_tuple = (imgs, img_metas, points[:, :3], points_label.astype(np.uint8))
        return data_tuple
    
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
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
        """
        # standard protocal modified from SECOND.Pytorch
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
        for cam_type, cam_info in info['cams'].items():
            image_paths.append(cam_info['data_path'])
            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
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

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
            ))

        return input_dict
    

def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]

    return nuScenes_label_name

class ImagePoint_FLINK(data.Dataset):
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
        self.dataset_loaders: dict[str, FlinkDatasetLoader] = {}

        REQUIRED_FOLDERS = {'depth', 'images', 'labels', 'metadata'}

        valid_dataset_paths: List[Path] = []
        def check_directory(dir_path: Path):
            # Check if this directory contains any of the required folders
            if any((dir_path / folder).exists() for folder in REQUIRED_FOLDERS):
                valid_dataset_paths.append(dir_path)
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
        
        if len_dataset is not None:
            self.len_dataset = len_dataset
        else:
            self.len_dataset = sum(len(loader) for loader in self.dataset_loaders.values())

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.len_dataset


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
        [selected_dataset_path, selected_dataset] = random.choice(list(self.dataset_loaders.items()))
        if selected_dataset is None:
            self.dataset_loaders[selected_dataset_path] = FlinkDatasetLoader(selected_dataset_path)
            selected_dataset = self.dataset_loaders[selected_dataset_path]
        # sample img_num images from the selected dataset
        datapoint_indices: List[int] = random.sample(range(len(selected_dataset)), self.img_num)
        selected_datapoints: List[FlinkDatapoint] = [selected_dataset[i] for i in datapoint_indices]

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
            imgs.append(datapoint.get_image().astype(np.float32))
            
            depth = datapoint.get_depth()
            
            points, valid_points = self._depth_to_pointcloud(depth, np.array(datapoint.metadata.metadata.camera_matrix), cam2lidar)
            labels = np.ones((depth.shape[0], depth.shape[1]), dtype=np.uint8)
            for segment in datapoint.label_data.segmentations:
                bbox = segment.bbox
                mask = self._rle_to_mask(segment.mask, (bbox[2], bbox[3]))
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
        if len(combined_points) > 20000:
            sample_indices = np.random.choice(len(combined_points), 20000, replace=False)
            combined_points = combined_points[sample_indices]
            combined_labels = combined_labels[sample_indices]

        img_metas: Dict[str, Any] = {
            'lidar2img': lidar2imgs,
        }  # Placeholder for consistency

        data_tuple: Tuple[List[np.ndarray], Dict[str, List[np.ndarray]], np.ndarray, np.ndarray] = (imgs, img_metas, combined_points, combined_labels)
        return data_tuple
    
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

    # @staticmethod
    # @nb.njit(parallel=True)
    # def _voxel_downsample(points: np.ndarray, voxel_size: float, labels: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
    #     """
    #     Simple voxel downsampling by randomly selecting one point per voxel.

    #     Args:
    #         points (np.ndarray): (n, 3) array of 3D points.
    #         voxel_size (float): Voxel size for downsampling.
    #         labels (np.ndarray | None, optional): Point labels. Defaults to None.

    #     Returns:
    #         Tuple[np.ndarray, np.ndarray | None]: Downsampled points and corresponding labels.
    #     """
    #     # Get voxel indices and hash
    #     voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    #     voxel_hash = (voxel_indices * np.array([1, 2048, 2048*2048])).sum(axis=1)
        
    #     # Sort by hash
    #     sort_idx = np.argsort(voxel_hash)
    #     sorted_hash = voxel_hash[sort_idx]
        
    #     # Find unique voxels
    #     diff = np.ones_like(sorted_hash, dtype=np.bool_)
    #     diff[:-1] = sorted_hash[1:] != sorted_hash[:-1]
    #     unique_mask = diff.nonzero()[0]
    #     num_voxels = len(unique_mask)
        
    #     # Randomly select points
    #     downsampled_points = np.zeros((num_voxels, 3), dtype=np.float32)
    #     downsampled_labels = np.zeros((num_voxels, labels.shape[1]), dtype=labels.dtype)
    #     for i in range(num_voxels):
    #         start_idx = unique_mask[i]
    #         end_idx = unique_mask[i + 1] if i < num_voxels - 1 else len(sorted_hash)
    #         random_idx = sort_idx[start_idx + np.random.randint(0, end_idx - start_idx)]
    #         downsampled_points[i] = points[random_idx]
    #         if labels is not None:
    #             downsampled_labels[i] = labels[random_idx]
            
    #     return downsampled_points, None if labels is None else downsampled_labels
