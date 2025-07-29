import argparse, torch, os
import shutil
import numpy as np
import mmcv
from mmcv import Config
from collections import OrderedDict
import pandas as pd

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# pio.renderers.default = "notebook"


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
    voxel_size=np.array([0.2, 0.2, 0.2], dtype=np.float32),  # voxel size in the real world
    pt_coords=None,       # voxel coordinates of point cloud
    pt_label=None,   # label of point cloud
    label_name=None,
    raw_points=None,
    raw_labels=None,
):
    w, h, z = voxels.shape
    pt_coords = pt_coords.astype(int)

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
    
    # Helper: Convert RGB array to hex color string
    def rgb_to_hex(rgb: np.ndarray) -> str:
        """Convert an RGB array to a hex color string."""
        return '#{:02x}{:02x}{:02x}'.format(*rgb)

    # Build a mapping from label index to color string
    label_to_color = {v: rgb_to_hex(colors[k]) for k, v in label_name.items()}

    # Create subplots
    num_plots = 4 if (raw_points is not None and raw_labels is not None) else 3
    fig = make_subplots(
        rows=1, cols=num_plots,
        specs=[[{'type': 'scatter3d'}] * num_plots],
        subplot_titles=("Occupancy", "Predicted Point Cloud", "Ground Truth Point Cloud", "Raw Point Cloud")[:num_plots]
    )

    # --- Occupancy Plot (Mode 0 equivalent) ---
    grid_coords_occ = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    # grid_coords_occ[grid_coords_occ[:, 3] == 1, 3] = 20  # Handle special case

    fov_voxels_occ = grid_coords_occ
    # fov_voxels_occ = grid_coords_occ[
    #     (grid_coords_occ[:, 3] > 0) & (grid_coords_occ[:, 3] < 20)
    # ]
    fov_voxels_occ = fov_voxels_occ[fov_voxels_occ[:, 3] != 0]

    occ_vox_dataframe = pd.DataFrame(fov_voxels_occ, columns=['x', 'y', 'z', 'label'])
    occ_vox_dataframe['label'] = occ_vox_dataframe['label'].astype(int)
    occ_vox_dataframe['label_name'] = occ_vox_dataframe['label'].map(label_name)

    scatter_occ = px.scatter_3d(
        occ_vox_dataframe,
        x='x', y='y', z='z',
        color='label_name',
        color_discrete_map=label_to_color,
    )

    for trace in scatter_occ.data:
        trace.marker.symbol = "square"
        trace.legendgroup = trace.name
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=1)

    # --- Predicted Point Cloud Plot (Mode 1 equivalent) ---
    indexes = pt_coords[:, 0] * h * z + pt_coords[:, 1] * z + pt_coords[:, 2]
    indexes, pt_index = np.unique(indexes, return_index=True)
    pred_pts = pred_pts[pt_index]
    grid_coords_pred = grid_coords[indexes]
    grid_coords_pred = np.vstack([grid_coords_pred.T, pred_pts.reshape(-1)]).T

    # Map predicted labels to colors
    grid_coords_pred = grid_coords_pred[grid_coords_pred[:, 3] != 0]

    pred_vox_dataframe = pd.DataFrame(grid_coords_pred, columns=['x', 'y', 'z', 'label'])
    pred_vox_dataframe['label'] = pred_vox_dataframe['label'].astype(int)
    pred_vox_dataframe['label_name'] = pred_vox_dataframe['label'].map(label_name)
    scatter_pred = px.scatter_3d(
        pred_vox_dataframe,
        x='x', y='y', z='z',
        color='label_name',
        color_discrete_map=label_to_color,
    )

    for trace in scatter_pred.data:
        trace.marker.symbol = "square"
        trace.legendgroup = trace.name
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=2)

    # --- Ground Truth Point Cloud Plot (Mode 2 equivalent) ---
    indexes = pt_coords[:, 0] * h * z + pt_coords[:, 1] * z + pt_coords[:, 2]
    indexes, pt_index = np.unique(indexes, return_index=True)
    gt_label = pt_label[pt_index]
    grid_coords_gt = grid_coords[indexes]
    grid_coords_gt = np.vstack([grid_coords_gt.T, gt_label.reshape(-1)]).T

    gt_vox_dataframe = pd.DataFrame(grid_coords_gt, columns=['x', 'y', 'z', 'label'])
    gt_vox_dataframe['label'] = gt_vox_dataframe['label'].astype(int)
    gt_vox_dataframe['label_name'] = gt_vox_dataframe['label'].map(label_name)
    scatter_gt = px.scatter_3d(
        gt_vox_dataframe,
        x='x', y='y', z='z',
        color='label_name',
        color_discrete_map=label_to_color,
    )

    for trace in scatter_gt.data:
        trace.marker.symbol = "square"
        trace.legendgroup = trace.name
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=3)

    # --- Raw Point Cloud Plot (Mode 3 equivalent) ---
    if raw_points is not None and raw_labels is not None:
        raw_vox_dataframe = pd.DataFrame(raw_points, columns=['x', 'y', 'z'])
        raw_vox_dataframe['label'] = raw_labels.astype(int).squeeze(-1)
        raw_vox_dataframe['label_name'] = raw_vox_dataframe['label'].map(label_name)
        scatter_raw = px.scatter_3d(
            raw_vox_dataframe,
            x='x', y='y', z='z',
            color="label_name",
            color_discrete_map=label_to_color,
        )

        for trace in scatter_raw.data:
            trace.marker.symbol = "circle"
            trace.marker.size = 2
            trace.legendgroup = trace.name
            fig.add_trace(trace, row=1, col=4)

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
        scene4=dict(bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(range=range_val),
                    yaxis=dict(range=range_val),
                    zaxis=dict(range=range_val),
                    aspectmode='cube'), # Important for equal aspect ratio
    )

    return fig


def main():
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
    parser.add_argument('--frame-idx', type=int, default=[0], nargs='+',
                        help='List of frame indices to visualize, corresponding to the order in pkl file.')
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
        ckpt = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        print(my_model.load_state_dict(revise_ckpt(ckpt)))
    my_model.eval()

    # prepare data
    from visualization.dataset import ImagePoint_NuScenes_vis, DatasetWrapper_NuScenes_vis, ImagePoint_FLINK_vis

    if args.vis_train:
        pkl_path = 'data/nuscenes_infos_train.pkl'
    else:
        pkl_path = 'data/nuscenes_infos_val.pkl'
    
    data_path = cfg.val_data_loader["data_path"]
    label_name = dataset_config["label_name"]

    if dataset_config["dataset_type"] == "ImagePoint_NuScenes":
        from nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)
        pt_dataset = ImagePoint_NuScenes_vis(
            data_path, imageset=pkl_path,
            nusc=nusc)
    elif dataset_config["dataset_type"] == "ImagePoint_FLINK":
        pt_dataset = ImagePoint_FLINK_vis(
            data_path, label_name=label_name)
    else:
        raise ValueError(f"Invalid dataset type: {dataset_config['dataset_type']}")

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
    label_name = dataset_config["label_name"]

    for index in args.frame_idx:
        print(f'processing frame {index}')
        batch_data, filelist, _, timestamp = dataset[index]
        imgs, img_metas, vox_label, pt_coords, pt_label = batch_data
        imgs = torch.from_numpy(np.stack([imgs]).astype(np.float32)).to(device)
        pt_coords = torch.from_numpy(np.stack([pt_coords]).astype(np.float32)).to(device)
        with torch.no_grad():
            outputs_vox, outputs_pts = my_model(img=imgs, img_metas=[img_metas], points=pt_coords.clone())
            print(outputs_vox.shape)
            print(outputs_pts.shape)
        
            predict_vox = torch.argmax(outputs_vox, dim=1) # bs, w, h, z
            predict_vox = predict_vox.squeeze(0).cpu().numpy() # w, h, z

            predict_pts = torch.argmax(outputs_pts, dim=1) # bs, n, 1, 1
            predict_pts = predict_pts.squeeze().cpu().numpy() # n

        voxel_origin = dataset_config['min_volume_space']
        voxel_max = dataset_config['max_volume_space']
        grid_size = cfg.grid_size
        resolution = np.array([(e - s) / l for e, s, l in zip(voxel_max, voxel_origin, grid_size)], dtype=np.float32)

        frame_dir = os.path.join(args.save_path, str(index))
        os.makedirs(frame_dir, exist_ok=True)
        
        for filename in filelist:            
            if os.path.exists(filename):
                shutil.copy(filename, os.path.join(frame_dir, os.path.basename(filename)))
            else:
                print(f"File {filename} does not exist")

        fig = draw(predict_vox,
            predict_pts,
            voxel_origin,
            resolution,
            pt_coords.squeeze(0).cpu().numpy(),
            pt_label.squeeze(-1),
            label_name,
            raw_points=img_metas['raw_points'] if 'raw_points' in img_metas else None,
            raw_labels=img_metas['raw_labels'] if 'raw_labels' in img_metas else None
            )

        fig.show()

if __name__ == "__main__":
    main()