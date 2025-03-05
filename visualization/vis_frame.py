import argparse, torch, os
import shutil
import numpy as np
import mmcv
from mmcv import Config
from collections import OrderedDict

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def revise_ckpt(state_dict):
    tmp_k = list(state_dict.keys())[0]
    if tmp_k.startswith('module.'):
        state_dict = OrderedDict(
            {k[7:]: v for k, v in state_dict.items()})
    return state_dict


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid


def draw(
    voxels,          # semantic occupancy predictions
    pred_pts,        # lidarseg predictions
    vox_origin,
    voxel_size=0.2,  # voxel size in the real world
    grid=None,       # voxel coordinates of point cloud
    pt_label=None,   # label of point cloud
    save_dirs=None,
    cam_positions=None,
    focal_positions=None,
    timestamp=None,
):
    w, h, z = voxels.shape
    grid = grid.astype(int)

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    # Define colors (from vis_scene.py)
    colors = np.array(
        [
            [255, 120,  50],  # barrier              orange
            [255, 192, 203],  # bicycle              pink
            [255, 255,   0],  # bus                  yellow
            [  0, 150, 245],  # car                  blue
            [  0, 255, 255],  # construction_vehicle cyan
            [255, 127,   0],  # motorcycle           dark orange
            [255,   0,   0],  # pedestrian           red
            [255, 240, 150],  # traffic_cone         light yellow
            [135,  60,   0],  # trailer              brown
            [160,  32, 240],  # truck                purple
            [255,   0, 255],  # driveable_surface    dark pink
            # [175,   0,  75, 255],       # other_flat           dark red
            [139, 137, 137],
            [ 75,   0,  75],  # sidewalk             dard purple
            [150, 240,  80],  # terrain              light green
            [230, 230, 250],  # manmade              white
            [  0, 175,   0],  # vegetation           green
            [  0, 255, 127],  # ego car              dark cyan
            [255,  99,  71],  # ego car
            [  0, 191, 255]   # ego car
        ]
    ).astype(np.uint8)

    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=("Occupancy", "Predicted Point Cloud", "Ground Truth Point Cloud")
    )

    # --- Occupancy Plot (Mode 0 equivalent) ---
    grid_coords_occ = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    grid_coords_occ[grid_coords_occ[:, 3] == 1, 3] = 20  # Handle special case

    fov_voxels_occ = grid_coords_occ[
        (grid_coords_occ[:, 3] > 0) & (grid_coords_occ[:, 3] < 20)
    ]
    fov_voxels_occ_colors = colors[fov_voxels_occ[:, 3].astype(int) % len(colors)]

    scatter_occ = go.Scatter3d(
        x=fov_voxels_occ[:, 1], y=fov_voxels_occ[:, 0], z=fov_voxels_occ[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=[f"rgb({r}, {g}, {b})" for r, g, b in fov_voxels_occ_colors],
            colorscale='Viridis',
            opacity=1.0,
            colorbar=dict(title="Occupancy", x=0.28)  # Adjust colorbar position
        )
    )
    fig.add_trace(scatter_occ, row=1, col=1)

    # --- Predicted Point Cloud Plot (Mode 1 equivalent) ---
    indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
    indexes, pt_index = np.unique(indexes, return_index=True)
    pred_pts = pred_pts[pt_index]
    grid_coords_pred = grid_coords[indexes]
    grid_coords_pred = np.vstack([grid_coords_pred.T, pred_pts.reshape(-1)]).T

    # Map predicted labels to colors
    pred_colors = colors[grid_coords_pred[:, 3].astype(int) % len(colors)]  # Handle out-of-bound labels

    scatter_pred = go.Scatter3d(
        x=grid_coords_pred[:, 1], y=grid_coords_pred[:, 0], z=grid_coords_pred[:, 2],
        mode='markers',
        marker=dict(
            size=voxel_size*5, # Use voxel_size for cube size
            color=[f"rgb({r}, {g}, {b})" for r, g, b in pred_colors],  # Use the mapped colors
            opacity=1.0,
            colorbar=dict(title="Predicted", x=0.64), # Adjust colorbar position
            symbol='square'  # Change marker symbol to square (cube in 3D)
        )
    )
    fig.add_trace(scatter_pred, row=1, col=2)

    # --- Ground Truth Point Cloud Plot (Mode 2 equivalent) ---
    indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
    indexes, pt_index = np.unique(indexes, return_index=True)
    gt_label = pt_label[pt_index]
    grid_coords_gt = grid_coords[indexes]
    grid_coords_gt = np.vstack([grid_coords_gt.T, gt_label.reshape(-1)]).T

    # Map ground truth labels to colors
    gt_colors = colors[grid_coords_gt[:, 3].astype(int) % len(colors)] # Handle out-of-bound labels

    scatter_gt = go.Scatter3d(
        x=grid_coords_gt[:, 1], y=grid_coords_gt[:, 0], z=grid_coords_gt[:, 2],
        mode='markers',
        marker=dict(
            size=voxel_size*5,  #Use voxel_size for cube size
            color=[f"rgb({r}, {g}, {b})" for r, g, b in gt_colors], # Use the mapped colors.
            opacity=1.0,
            colorbar=dict(title="Ground Truth", x=1.0),  # Adjust colorbar position
            symbol='square' # Change marker symbol to square (cube in 3D)
        )
    )
    fig.add_trace(scatter_gt, row=1, col=3)

    # --- Layout Settings ---
    # Calculate axis ranges based on your data's extent
    min_x, max_x = grid_coords[:, 1].min(), grid_coords[:, 1].max()
    min_y, max_y = grid_coords[:, 0].min(), grid_coords[:, 0].max()
    min_z, max_z = grid_coords[:, 2].min(), grid_coords[:, 2].max()

    # Find overall min and max to create a common range
    overall_min = min(min_x, min_y, min_z)
    overall_max = max(max_x, max_y, max_z)
    range_val = [overall_min, overall_max]

    fig.update_layout(
        title="TPVFormer Visualization",
        height=800, width=2400,  # Adjusted width for three plots
        scene=dict(bgcolor="rgba(0,0,0,0)",
                   xaxis=dict(range=range_val),
                   yaxis=dict(range=range_val),
                   zaxis=dict(range=range_val),
                   aspectmode='cube'),  # Important for equal aspect ratio
        scene2=dict(bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(range=range_val),
                    yaxis=dict(range=range_val),
                    zaxis=dict(range=range_val),
                    aspectmode='cube'), # Important for equal aspect ratio
        scene3=dict(bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(range=range_val),
                    yaxis=dict(range=range_val),
                    zaxis=dict(range=range_val),
                    aspectmode='cube'), # Important for equal aspect ratio
        showlegend=False
    )

    fig.show()


if __name__ == "__main__":
    import sys; sys.path.insert(0, os.path.abspath('.'))

    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    ## prepare config
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv04_occupancy.py')
    parser.add_argument('--work-dir', type=str, default='out/tpv_occupancy')
    parser.add_argument('--ckpt-path', type=str, default='out/tpv_occupancy/latest.pth')
    parser.add_argument('--vis-train', action='store_true', default=False)
    parser.add_argument('--save-path', type=str, default='out/tpv_occupancy/frames')
    parser.add_argument('--frame-idx', type=int, default=0, nargs='+', 
                        help='idx of frame to visualize, the idx corresponds to the order in pkl file.')
    # parser.add_argument('--mode', type=int, default=0, help='0: occupancy, 1: predicted point cloud, 2: gt point cloud') # Removed mode argument
    args = parser.parse_args()
    print(args)


    cfg = Config.fromfile(args.py_config)
    dataset_config = cfg.dataset_params

    # prepare model
    logger = mmcv.utils.get_logger('mmcv')
    logger.setLevel("WARNING")
    if cfg.get('occupancy', False):
        from builder import tpv_occupancy_builder as model_builder
    else:
        from builder import tpv_lidarseg_builder as model_builder
    my_model = model_builder.build(cfg.model).to(device)
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        print(my_model.load_state_dict(revise_ckpt(ckpt)))
    my_model.eval()

    # prepare data
    from nuscenes import NuScenes
    from visualization.dataset import ImagePoint_NuScenes_vis, DatasetWrapper_NuScenes_vis, ImagePoint_FLINK_vis

    if args.vis_train:
        pkl_path = 'data/nuscenes_infos_train.pkl'
    else:
        pkl_path = 'data/nuscenes_infos_val.pkl'
    
    data_path = cfg.train_data_loader["data_path"]
    label_mapping = dataset_config['label_mapping']

    if dataset_config["dataset_type"] == "ImagePoint_NuScenes":
        nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)
            
        pt_dataset = ImagePoint_NuScenes_vis(
            data_path, imageset=pkl_path,
            label_mapping=label_mapping, nusc=nusc)

        dataset = DatasetWrapper_NuScenes_vis(
            pt_dataset,
            grid_size=cfg.grid_size,
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            ignore_label=dataset_config["fill_label"],
            phase='val'
        )
        print(len(dataset))
    elif dataset_config["dataset_type"] == "ImagePoint_FLINK":
        pt_dataset = ImagePoint_FLINK_vis(
            data_path, label_mapping=label_mapping)

        dataset = DatasetWrapper_NuScenes_vis(
            pt_dataset,
            grid_size=cfg.grid_size,
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            ignore_label=dataset_config["fill_label"],
            phase='val'
        )
        print(len(dataset))
    else:
        raise ValueError(f"Invalid dataset type: {dataset_config['dataset_type']}")


    for index in args.frame_idx:
        print(f'processing frame {index}')
        batch_data, filelist, _, timestamp = dataset[index]
        imgs, img_metas, vox_label, grid, pt_label = batch_data
        imgs = torch.from_numpy(np.stack([imgs]).astype(np.float32)).to(device)
        grid = torch.from_numpy(np.stack([grid]).astype(np.float32)).to(device)
        with torch.no_grad():
            outputs_vox, outputs_pts = my_model(img=imgs, img_metas=[img_metas], points=grid.clone())
        
            predict_vox = torch.argmax(outputs_vox, dim=1) # bs, w, h, z
            predict_vox = predict_vox.squeeze(0).cpu().numpy() # w, h, z

            predict_pts = torch.argmax(outputs_pts, dim=1) # bs, n, 1, 1
            predict_pts = predict_pts.squeeze().cpu().numpy() # n

        voxel_origin = dataset_config['min_volume_space']
        voxel_max = dataset_config['max_volume_space']
        grid_size = cfg.grid_size
        resolution = [(e - s) / l for e, s, l in zip(voxel_max, voxel_origin, grid_size)]

        frame_dir = os.path.join(args.save_path, str(index))
        os.makedirs(frame_dir, exist_ok=True)
        
        for filename in filelist:            
            if os.path.exists(filename):
                shutil.copy(filename, os.path.join(frame_dir, os.path.basename(filename)))
            else:
                print(f"File {filename} does not exist")

        draw(predict_vox,
            predict_pts,
            voxel_origin,
            resolution,
            grid.squeeze(0).cpu().numpy(),
            pt_label.squeeze(-1),
            frame_dir,
            img_metas['cam_positions'],
            img_metas['focal_positions'],
            timestamp=timestamp)

