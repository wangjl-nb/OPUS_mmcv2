import argparse
import copy

import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize GT and prediction PLY together with Open3D')
    parser.add_argument('--gt', required=True, help='Path to GT PLY file')
    parser.add_argument('--pred', required=True, help='Path to prediction PLY file')
    parser.add_argument('--mode', choices=['side-by-side', 'overlay'],
                        default='side-by-side', help='Comparison layout mode')
    parser.add_argument('--gap', type=float, default=1.0,
                        help='Gap between GT and prediction in side-by-side mode')
    parser.add_argument('--max-points', type=int, default=300000,
                        help='Randomly keep at most this many points per cloud; <=0 keeps all')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for point sampling')
    parser.add_argument('--paint-gt', nargs=3, type=float, default=None,
                        metavar=('R', 'G', 'B'), help='Override GT color, values in [0, 1]')
    parser.add_argument('--paint-pred', nargs=3, type=float, default=None,
                        metavar=('R', 'G', 'B'), help='Override prediction color, values in [0, 1]')
    parser.add_argument('--point-size', type=float, default=1.5,
                        help='Open3D point size')
    parser.add_argument('--bg', choices=['black', 'white'], default='black',
                        help='Viewer background color')
    parser.add_argument('--show-axis', action='store_true',
                        help='Show coordinate frame')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only print cloud stats without opening visualization window')
    return parser.parse_args()


def load_cloud(path):
    cloud = o3d.io.read_point_cloud(path)
    if cloud.is_empty():
        raise ValueError(f'Failed to read points from: {path}')
    return cloud


def maybe_subsample(cloud, max_points, rng):
    if max_points is None or max_points <= 0:
        return cloud
    n_points = len(cloud.points)
    if n_points <= max_points:
        return cloud
    indices = rng.choice(n_points, size=max_points, replace=False)
    return cloud.select_by_index(indices.tolist())


def cloud_stats(name, cloud):
    center = cloud.get_center()
    extent = cloud.get_axis_aligned_bounding_box().get_extent()
    print(
        f'{name}: points={len(cloud.points)}, '
        f'center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}), '
        f'extent=({extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f})'
    )


def prepare_side_by_side(gt_cloud, pred_cloud, gap):
    gt_vis = copy.deepcopy(gt_cloud)
    pred_vis = copy.deepcopy(pred_cloud)

    gt_vis.translate(-gt_vis.get_center())
    pred_vis.translate(-pred_vis.get_center())

    gt_extent_x = gt_vis.get_axis_aligned_bounding_box().get_extent()[0]
    pred_extent_x = pred_vis.get_axis_aligned_bounding_box().get_extent()[0]
    shift = max(gt_extent_x, pred_extent_x) + gap

    gt_vis.translate(np.array([-0.5 * shift, 0.0, 0.0]))
    pred_vis.translate(np.array([0.5 * shift, 0.0, 0.0]))
    return gt_vis, pred_vis


def build_separator(gt_cloud, pred_cloud):
    all_points = np.concatenate([
        np.asarray(gt_cloud.points),
        np.asarray(pred_cloud.points)
    ], axis=0)
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    z_mid = 0.5 * (z_min + z_max)

    line_points = np.array([[0.0, y_min, z_mid], [0.0, y_max, z_mid]], dtype=np.float64)
    line_indices = np.array([[0, 1]], dtype=np.int32)
    line_colors = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)

    separator = o3d.geometry.LineSet()
    separator.points = o3d.utility.Vector3dVector(line_points)
    separator.lines = o3d.utility.Vector2iVector(line_indices)
    separator.colors = o3d.utility.Vector3dVector(line_colors)
    return separator


def maybe_paint(cloud, color):
    if color is None:
        return
    cloud.paint_uniform_color(np.asarray(color, dtype=np.float64))


def visualize(geometries, args):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='GT vs Pred (Open3D)', width=1600, height=900)

    for geometry in geometries:
        vis.add_geometry(geometry)

    render = vis.get_render_option()
    render.point_size = float(args.point_size)
    render.background_color = np.array([0.0, 0.0, 0.0]) if args.bg == 'black' else np.array([1.0, 1.0, 1.0])
    render.show_coordinate_frame = bool(args.show_axis)

    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    gt_cloud = load_cloud(args.gt)
    pred_cloud = load_cloud(args.pred)

    gt_cloud = maybe_subsample(gt_cloud, args.max_points, rng)
    pred_cloud = maybe_subsample(pred_cloud, args.max_points, rng)

    maybe_paint(gt_cloud, args.paint_gt)
    maybe_paint(pred_cloud, args.paint_pred)

    cloud_stats('GT', gt_cloud)
    cloud_stats('Pred', pred_cloud)

    if args.mode == 'side-by-side':
        gt_vis, pred_vis = prepare_side_by_side(gt_cloud, pred_cloud, args.gap)
        separator = build_separator(gt_vis, pred_vis)
        geometries = [gt_vis, pred_vis, separator]
        print('Mode: side-by-side (left=GT, right=Pred)')
    else:
        geometries = [gt_cloud, pred_cloud]
        print('Mode: overlay (GT and Pred in the same coordinates)')

    if args.dry_run:
        return

    visualize(geometries, args)


if __name__ == '__main__':
    main()
