import numpy as np
import cv2
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import open3d as o3d


# ============================================================================
# Configuration
# ============================================================================

# Dataset configuration
SCENE = "Office"
FRAME = 779
BASE_DIR = "/media/baai/7E357B68494ABB7B/model/map-anything/mapanything"
SCENE_DIR = f"{BASE_DIR}/tartanground/{SCENE}/Data_omni/P0001"
OUTPUT_DIR = f"{BASE_DIR}/reconstruction_output"

# Camera configuration
VIEWS = ["front", "left", "back", "right", "top", "bottom"]
INTRINSICS = np.array([
    [320, 0, 320],
    [0, 320, 320],
    [0, 0, 1]
], dtype=np.float32)

# DINOv2 model configuration
DINOV2_REPO = "/home/baai/.cache/torch/hub/facebookresearch_dinov2_main"
DINOV2_MODEL = "dinov2_vitl14"

# Processing configuration
DEFAULT_VOXEL_SIZE = 0.01
MAX_DEPTH = 50.0


# ============================================================================
# Utility Functions
# ============================================================================

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_output_path(filename):
    """Get full path for output file."""
    ensure_output_dir()
    return os.path.join(OUTPUT_DIR, filename)


def load_view_images(view_name, frame=FRAME, scene_dir=SCENE_DIR):
    """
    Load depth and RGB images for a specific view.

    Args:
        view_name: Name of the view (front, left, back, right, top, bottom)
        frame: Frame number
        scene_dir: Scene directory path

    Returns:
        depth: Depth image (H, W) or None if failed
        rgb: RGB image (H, W, 3) or None if failed
    """
    depth_path = os.path.join(scene_dir, f"depth_lcam_{view_name}", f"{frame:06d}_lcam_{view_name}_depth.png")
    rgb_path = os.path.join(scene_dir, f"image_lcam_{view_name}", f"{frame:06d}_lcam_{view_name}.png")

    depth = load_depth(depth_path)
    rgb = cv2.imread(rgb_path)

    if rgb is not None:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    return depth, rgb


def load_dinov2_model(device=None):
    """
    Load DINOv2 model.

    Args:
        device: torch device. If None, auto-detect cuda/cpu

    Returns:
        model: DINOv2 model
        device: torch device used
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading DINOv2 model on {device}...")
    model = torch.hub.load(
        repo_or_dir=DINOV2_REPO,
        model=DINOV2_MODEL,
        trust_repo=True,
        source="local"
    )
    model = model.to(device).eval()
    print("DINOv2 model loaded successfully!")

    return model, device


def load_depth(depth_path):
    """
    Load depth image from PNG or NPY file.

    Args:
        depth_path (str): Path to depth image file

    Returns:
        np.ndarray or None: Depth image as numpy array, None if failed to load
    """
    if depth_path.endswith('.npy'):
        return np.load(depth_path)

    depth_rgba = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_rgba is None:
        return None
    return depth_rgba.view("<f4").squeeze()


def extract_dinov2_features(rgb_image, model, device, patch_size=14):
    """
    Extract DINOv2 features for an RGB image and interpolate to pixel resolution.

    Args:
        rgb_image: RGB image (H, W, 3) in range [0, 255]
        model: DINOv2 model
        device: torch device
        patch_size: Patch size of the model (14 for ViT-L/14)

    Returns:
        features: Per-pixel features (H, W, feature_dim)
    """
    orig_h, orig_w = rgb_image.shape[:2]

    # Adjust size to be divisible by patch_size
    new_h = (orig_h // patch_size) * patch_size
    new_w = (orig_w // patch_size) * patch_size

    if new_h == 0:
        new_h = patch_size
    if new_w == 0:
        new_w = patch_size

    # Prepare transform
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((new_h, new_w)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(rgb_image).unsqueeze(0).to(device)

    # Extract patch features
    with torch.no_grad():
        patch_features = model.get_intermediate_layers(img_tensor, n=1)[0]
        # patch_features shape: [1, num_patches, feature_dim]

    batch_size, num_patches, feature_dim = patch_features.shape
    patch_h = new_h // patch_size
    patch_w = new_w // patch_size

    # Reshape to spatial dimensions [1, feature_dim, patch_h, patch_w]
    patch_features_spatial = patch_features.reshape(1, patch_h, patch_w, feature_dim).permute(0, 3, 1, 2)

    # Interpolate to original image size
    features_upsampled = F.interpolate(
        patch_features_spatial,
        size=(orig_h, orig_w),
        mode='bilinear',
        align_corners=False
    )

    # Convert to numpy: [1, feature_dim, H, W] -> [H, W, feature_dim]
    features_np = features_upsampled[0].permute(1, 2, 0).cpu().numpy()

    return features_np


def depth_to_pointcloud(depth, rgb, intrinsics, extrinsics, max_depth=50.0, features=None):
    """
    Convert depth image to 3D point cloud with RGB colors and optional features.

    Args:
        depth: Depth image (H, W)
        rgb: RGB image (H, W, 3)
        intrinsics: Camera intrinsic matrix (3, 3)
        extrinsics: Camera extrinsic matrix (4, 4) - transformation from camera to world
        max_depth: Maximum depth value to include (in meters)
        features: Optional per-pixel features (H, W, feature_dim)

    Returns:
        points: 3D points in world coordinates (N, 3)
        colors: RGB colors (N, 3)
        point_features: Features for each point (N, feature_dim) or None
    """
    h, w = depth.shape

    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten()
    v = v.flatten()
    depth_flat = depth.flatten()

    # Filter out invalid depth values
    valid_mask = (depth_flat > 0) & (depth_flat < max_depth)
    u = u[valid_mask]
    v = v[valid_mask]
    depth_flat = depth_flat[valid_mask]

    # Get intrinsic parameters
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Convert to 3D points in camera coordinates
    x_cam = (u - cx) * depth_flat / fx
    y_cam = (v - cy) * depth_flat / fy
    z_cam = depth_flat

    # Stack to homogeneous coordinates
    points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(x_cam)], axis=1)  # (N, 4)

    # Transform to world coordinates
    points_world = points_cam @ extrinsics  # (N, 4)
    points_world = points_world[:, :3]  # (N, 3)

    # Get corresponding colors
    colors = rgb[v, u, :]  # (N, 3)

    # Get corresponding features if provided
    point_features = None
    if features is not None:
        point_features = features[v, u, :]  # (N, feature_dim)

    return points_world, colors, point_features


def voxel_downsample(points, colors, voxel_size=0.05, features=None):
    """
    Downsample point cloud using voxel grid method with Open3D.

    Args:
        points: 3D points (N, 3)
        colors: RGB colors (N, 3)
        voxel_size: Size of voxel grid for downsampling
        features: Optional per-point features (N, feature_dim)

    Returns:
        downsampled_points: Downsampled 3D points
        downsampled_colors: Downsampled colors
        downsampled_features: Downsampled features (if features provided)
    """
    # Use Open3D for voxel downsampling with trace_down to get indices
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Open3D expects colors in [0, 1]

    # Voxel downsampling with index tracking
    pcd_down, _, indices_list = pcd.voxel_down_sample_and_trace(
        voxel_size=voxel_size,
        min_bound=pcd.get_min_bound() - voxel_size * 0.5,
        max_bound=pcd.get_max_bound() + voxel_size * 0.5
    )

    downsampled_points = np.asarray(pcd_down.points)
    downsampled_colors = np.asarray(pcd_down.colors) * 255.0  # Convert back to [0, 255]

    downsampled_features = None
    if features is not None:
        # Average features within each voxel
        downsampled_features = np.zeros((len(indices_list), features.shape[1]))
        for i, indices in enumerate(indices_list):
            idx_array = np.asarray(indices)
            if len(idx_array) > 0:
                downsampled_features[i] = features[idx_array].mean(axis=0)

    return downsampled_points, downsampled_colors, downsampled_features


def get_camera_extrinsics(view_name):
    """
    Get camera extrinsic matrix for each view.
    All cameras are positioned at origin, only rotations differ.

    Camera coordinate system: X-right, Y-down, Z-forward (depth direction)

    Args:
        view_name: Name of the view (front, left, back, right, top, bottom)

    Returns:
        extrinsics: 4x4 transformation matrix from camera to reference (front) coordinates
    """
    # Initialize as identity
    extrinsics = np.eye(4)

    if view_name == "front":
        # Front view: reference frame, no rotation
        R = np.eye(3)
    elif view_name == "left":
        R = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
    elif view_name == "back":
        R = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
    elif view_name == "right":
        R = np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0]
        ])
    elif view_name == "top":
        R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
    elif view_name == "bottom":
        R = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
    else:
        raise ValueError(f"Unknown view name: {view_name}")

    extrinsics[:3, :3] = R
    # Translation is zero (all cameras at origin)
    extrinsics[:3, 3] = [0, 0, 0]

    return extrinsics


def save_pointcloud_ply(filename, points, colors):
    """
    Save point cloud to PLY file using Open3D.

    Args:
        filename: Output PLY file path
        points: 3D points (N, 3)
        colors: RGB colors (N, 3), values in range [0, 255]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Open3D expects colors in [0, 1]

    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud with {len(points)} points to {filename}")


def reconstruct_scene(debug=False, voxel_size=DEFAULT_VOXEL_SIZE, extract_features=False):
    """
    Reconstruct 3D point cloud from 6 views.

    Args:
        debug: If True, save individual point clouds for each view
        voxel_size: Voxel size for downsampling
        extract_features: If True, extract DINOv2 features and save feature-colored point cloud
    """
    # Load DINOv2 model if needed
    dinov2_model = None
    device = None
    if extract_features:
        dinov2_model, device = load_dinov2_model()

    # Collect all points, colors, and features
    all_points = []
    all_colors = []
    all_features = [] if extract_features else None

    print("\nProcessing views...")
    for view in VIEWS:
        print(f"  Processing {view} view...")

        # Load depth and RGB images
        depth, rgb = load_view_images(view)

        if depth is None or rgb is None:
            print(f"    Warning: Failed to load images for {view} view")
            continue

        # Extract DINOv2 features if needed
        features = None
        if extract_features:
            print(f"    Extracting DINOv2 features...")
            features = extract_dinov2_features(rgb, dinov2_model, device)
            print(f"    Feature shape: {features.shape}")

        # Get camera extrinsics
        extrinsics = get_camera_extrinsics(view)
        if not extract_features:
            print(f"    Rotation matrix for {view}:")
            print(f"    {extrinsics[:3, :3]}")

        # Convert to point cloud
        points, colors, point_features = depth_to_pointcloud(
            depth, rgb, INTRINSICS, extrinsics, max_depth=MAX_DEPTH, features=features
        )

        print(f"    Generated {len(points)} points from {view} view")
        if not extract_features:
            print(f"    Point cloud bounds: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
                  f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
                  f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

        # Save individual view if debug mode
        if debug:
            debug_path = get_output_path(f"debug_{view}.ply")
            save_pointcloud_ply(debug_path, points, colors)

        all_points.append(points)
        all_colors.append(colors)
        if extract_features:
            all_features.append(point_features)

    # Merge all point clouds
    print("\nMerging point clouds...")
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)
    if extract_features:
        all_features = np.vstack(all_features)
        print(f"Feature dimension: {all_features.shape[1]}")

    print(f"Total points before downsampling: {len(all_points)}")
    if not extract_features:
        print(f"Merged point cloud bounds: X[{all_points[:, 0].min():.2f}, {all_points[:, 0].max():.2f}], "
              f"Y[{all_points[:, 1].min():.2f}, {all_points[:, 1].max():.2f}], "
              f"Z[{all_points[:, 2].min():.2f}, {all_points[:, 2].max():.2f}]")

    # Apply voxel downsampling
    print("\nApplying voxel downsampling...")
    all_points, all_colors, all_features = voxel_downsample(
        all_points, all_colors, voxel_size=voxel_size, features=all_features
    )
    print(f"Total points after downsampling: {len(all_points)}")

    # Save point cloud with RGB colors
    output_path = get_output_path("reconstructed_scene.ply")
    save_pointcloud_ply(output_path, all_points, all_colors)
    print(f"Saved RGB point cloud to: {output_path}")

    # Process features if extracted
    if extract_features and all_features is not None:
        # Standardize features before PCA
        print("\nStandardizing features...")
        scaler = StandardScaler()
        all_features_scaled = scaler.fit_transform(all_features)
        print(f"Feature mean: {all_features_scaled.mean(axis=0)[:5]}")  # Should be ~0
        print(f"Feature std: {all_features_scaled.std(axis=0)[:5]}")    # Should be ~1

        # Apply PCA to reduce features to 3 dimensions
        print("\nApplying PCA to reduce features to 3D...")
        pca = PCA(n_components=3, random_state=0)
        pca_features = pca.fit_transform(all_features_scaled)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

        # Normalize PCA features to [0, 255] for RGB
        pca_min = pca_features.min(axis=0)
        pca_max = pca_features.max(axis=0)
        pca_normalized = (pca_features - pca_min) / (pca_max - pca_min + 1e-8)
        feature_colors = (pca_normalized * 255).astype(np.uint8)

        # Save point cloud with PCA feature colors
        output_feature_path = get_output_path("reconstructed_scene_features.ply")
        save_pointcloud_ply(output_feature_path, all_points, feature_colors)
        print(f"Saved feature-colored point cloud to: {output_feature_path}")

        # Save the raw features as numpy file for later use
        feature_output_path = get_output_path("pointcloud_features.npz")
        np.savez(
            feature_output_path,
            points=all_points,
            colors=all_colors,
            features=all_features,
            pca_features=pca_features,
            feature_colors=feature_colors
        )
        print(f"Saved raw features to: {feature_output_path}")

    print(f"\nReconstruction complete!")


def visualize_feature_similarity(view_name, pixel_x, pixel_y, feature_npz_path=None):
    """
    Visualize feature similarity by selecting a pixel from a specific view.
    Points with similar features are colored red, dissimilar ones are blue.
    If feature file doesn't exist, will reconstruct with features first.

    Args:
        view_name: Name of the view (front, left, back, right, top, bottom)
        pixel_x: X coordinate of the selected pixel
        pixel_y: Y coordinate of the selected pixel
        feature_npz_path: Path to the saved feature npz file. If None, uses default path.
    """
    # Check if feature file exists
    if feature_npz_path is None:
        feature_npz_path = get_output_path("pointcloud_features.npz")

    if not os.path.exists(feature_npz_path):
        print(f"Feature file not found at {feature_npz_path}")
        print("Reconstructing scene with features first...")
        reconstruct_scene(debug=False, voxel_size=DEFAULT_VOXEL_SIZE, extract_features=True)
        print("\nNow computing similarity visualization...\n")

    # Load saved features
    print(f"Loading features from {feature_npz_path}...")
    data = np.load(feature_npz_path)
    points = data['points']
    features = data['features']

    print(f"Loaded {len(points)} points with {features.shape[1]}-dimensional features")

    # Load RGB and depth for the selected view
    depth, rgb = load_view_images(view_name)

    if depth is None or rgb is None:
        print(f"Error: Failed to load images for {view_name} view")
        return

    # Check if pixel is valid
    h, w = rgb.shape[:2]
    if pixel_x < 0 or pixel_x >= w or pixel_y < 0 or pixel_y >= h:
        print(f"Error: Pixel ({pixel_x}, {pixel_y}) is out of bounds for image size ({w}, {h})")
        return

    # Check if pixel has valid depth
    if depth[pixel_y, pixel_x] <= 0:
        print(f"Error: Selected pixel ({pixel_x}, {pixel_y}) has invalid depth")
        return

    # Extract features for this view
    print(f"\nExtracting features for {view_name} view...")
    dinov2_model, device = load_dinov2_model()
    view_features = extract_dinov2_features(rgb, dinov2_model, device)

    # Get the query feature at the selected pixel
    query_feature = view_features[pixel_y, pixel_x, :]  # (feature_dim,)
    print(f"Query feature shape: {query_feature.shape}")
    print(f"Selected pixel RGB: {rgb[pixel_y, pixel_x]}")

    # Normalize features for cosine similarity
    query_feature_norm = query_feature / (np.linalg.norm(query_feature) + 1e-8)
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity
    print("\nComputing feature similarities...")
    similarities = np.dot(features_norm, query_feature_norm)  # (N,)

    print(f"Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")
    print(f"Similarity mean: {similarities.mean():.4f}, std: {similarities.std():.4f}")

    # Normalize similarities to [0, 1]
    sim_min = similarities.min()
    sim_max = similarities.max()
    similarities_normalized = (similarities - sim_min) / (sim_max - sim_min + 1e-8)

    # Create color map: blue (low similarity) -> red (high similarity)
    # Blue: (0, 0, 255), Red: (255, 0, 0)
    similarity_colors = np.zeros((len(points), 3), dtype=np.uint8)
    similarity_colors[:, 0] = (similarities_normalized * 255).astype(np.uint8)  # Red channel
    similarity_colors[:, 2] = ((1 - similarities_normalized) * 255).astype(np.uint8)  # Blue channel

    # Save similarity-colored point cloud
    output_path = get_output_path(f"similarity_{view_name}_x{pixel_x}_y{pixel_y}.ply")
    save_pointcloud_ply(output_path, points, similarity_colors)

    print(f"\nSimilarity visualization saved to: {output_path}")
    print(f"Red = high similarity, Blue = low similarity")

    # Also save a version with top-k most similar points highlighted
    k = int(len(points) * 0.05)  # Top 5% most similar points
    top_k_indices = np.argsort(similarities)[-k:]

    highlight_colors = np.zeros((len(points), 3), dtype=np.uint8)
    highlight_colors[:] = [128, 128, 128]  # Gray for all points
    highlight_colors[top_k_indices] = [255, 0, 0]  # Red for top-k similar points

    output_topk_path = get_output_path(f"similarity_topk_{view_name}_x{pixel_x}_y{pixel_y}.ply")
    save_pointcloud_ply(output_topk_path, points, highlight_colors)
    print(f"Top {k} similar points highlighted in: {output_topk_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reconstruct 3D point cloud from multi-view images")

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Reconstruct command
    reconstruct_parser = subparsers.add_parser('reconstruct', help='Reconstruct point cloud from multi-view images')
    reconstruct_parser.add_argument("--extract-features", action="store_true", help="Extract DINOv2 features and save feature-colored point cloud")
    reconstruct_parser.add_argument("--debug", action="store_true", help="Save individual point clouds for each view")
    reconstruct_parser.add_argument("--voxel-size", type=float, default=0.1, help="Voxel size for downsampling")

    # Similarity command
    similarity_parser = subparsers.add_parser('similarity', help='Visualize feature similarity from a selected pixel')
    similarity_parser.add_argument("--view", type=str, required=True, choices=["front", "left", "back", "right", "top", "bottom"], help="View name")
    similarity_parser.add_argument("--x", type=int, required=True, help="Pixel X coordinate")
    similarity_parser.add_argument("--y", type=int, required=True, help="Pixel Y coordinate")
    similarity_parser.add_argument("--feature-file", type=str, default=None, help="Path to feature npz file (default: pointcloud_features.npz)")

    args = parser.parse_args()

    if args.command == 'reconstruct':
        reconstruct_scene(debug=args.debug, voxel_size=args.voxel_size, extract_features=args.extract_features)
    elif args.command == 'similarity':
        visualize_feature_similarity(args.view, args.x, args.y, args.feature_file)
    else:
        # Default behavior: reconstruct without features
        reconstruct_scene(debug=False, voxel_size=0.1, extract_features=False)
