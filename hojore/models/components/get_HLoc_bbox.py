import os
import subprocess
import numpy as np
import trimesh
import cv2
from colmap_readmodel import read_cameras_binary, read_images_text

def run_colmap(image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    database_path = os.path.join(output_folder, 'database.db')
    sparse_folder = os.path.join(output_folder, 'sparse')
    dense_folder = os.path.join(output_folder, 'dense')
    os.makedirs(sparse_folder, exist_ok=True)
    os.makedirs(dense_folder, exist_ok=True)

    # Feature extraction
    subprocess.run([
        'colmap', 'feature_extractor',
        '--database_path', database_path,
        '--image_path', image_folder,
        '--ImageReader.single_camera', '1'
    ])

    # Feature matching
    subprocess.run([
        'colmap', 'exhaustive_matcher',
        '--database_path', database_path
    ])

    # Structure from Motion (SfM)
    subprocess.run([
        'colmap', 'mapper',
        '--database_path', database_path,
        '--image_path', image_folder,
        '--output_path', sparse_folder
    ])

    # Convert model to PLY
    subprocess.run([
        'colmap', 'model_converter',
        '--input_path', os.path.join(sparse_folder, '0'),
        '--output_path', os.path.join(sparse_folder, '0'),
        '--output_type', 'PLY'
    ])

    # Dense reconstruction
    subprocess.run([
        'colmap', 'image_undistorter',
        '--image_path', image_folder,
        '--input_path', os.path.join(sparse_folder, '0'),
        '--output_path', dense_folder,
        '--output_type', 'COLMAP',
        '--max_image_size', '2000'
    ])

    subprocess.run([
        'colmap', 'patch_match_stereo',
        '--workspace_path', dense_folder,
        '--workspace_format', 'COLMAP',
        '--PatchMatchStereo.geom_consistency', 'true'
    ])

    subprocess.run([
        'colmap', 'stereo_fusion',
        '--workspace_path', dense_folder,
        '--workspace_format', 'COLMAP',
        '--input_type', 'geometric',
        '--output_path', os.path.join(dense_folder, 'fused.ply')
    ])

def load_point_cloud(filepath):
    mesh = trimesh.load(filepath, process=False)
    vertices = np.array(mesh.vertices)
    return vertices

def get_bounding_box(vertices):
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    return bbox_min, bbox_max

def project_3d_to_2d(points_3d, K, R, t):
    points_3d_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    extrinsics = np.hstack((R, t.reshape(-1, 1)))
    points_2d_h = K @ (extrinsics @ points_3d_h.T)
    points_2d = points_2d_h[:2, :] / points_2d_h[2, :]
    return points_2d.T

def draw_bounding_box(image_path, bbox_min, bbox_max, K, R, t, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    vertices = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
    ])

    bbox_2d = project_3d_to_2d(vertices, K, R, t)
    x_min, y_min = np.min(bbox_2d, axis=0).astype(int)
    x_max, y_max = np.max(bbox_2d, axis=0).astype(int)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imwrite(output_path, image)
    print(f"Output image with bounding box saved to {output_path}")

def main(image_folder, output_folder, sample_image_path, output_image_path):
    run_colmap(image_folder, output_folder)
    
    point_cloud_path = os.path.join(output_folder, 'dense', 'fused.ply')
    vertices = load_point_cloud(point_cloud_path)
    
    bbox_min, bbox_max = get_bounding_box(vertices)
    
    camera_data = read_cameras_binary(os.path.join(output_folder, 'sparse', '0', 'cameras.bin'))
    image_data = read_images_text(os.path.join(output_folder, 'sparse', '0', 'images.txt'))
    
    camera = next(iter(camera_data.values()))
    K = np.array([
        [camera.params[0], 0, camera.params[2]],
        [0, camera.params[1], camera.params[3]],
        [0, 0, 1]
    ])
    
    image_name = os.path.basename(sample_image_path)
    for img in image_data.values():
        if img.name == image_name:
            R = img.qvec2rotmat()
            t = img.tvec
            break
    else:
        raise ValueError("Sample image not found in COLMAP output")

    draw_bounding_box(sample_image_path, bbox_min, bbox_max, K, R, t, output_image_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process images with COLMAP and extract bounding box for an object.")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing input images")
    parser.add_argument("output_folder", type=str, help="Path to the folder to store COLMAP outputs")
    parser.add_argument("sample_image_path", type=str, help="Path to a sample image to draw the bounding box")
    parser.add_argument("output_image_path", type=str, help="Path to save the output image with bounding box")

    args = parser.parse_args()
    
    main(args.image_folder, args.output_folder, args.sample_image_path, args.output_image_path)
