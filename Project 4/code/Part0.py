import cv2
import numpy as np
import glob
import os
import json
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")

def calibrate_camera(images_path, tag_size=0.06, output_file="camera_calibration.npz"):
    """
    Calibrate camera using ArUco tags with CUDA acceleration if available
    """
    # Set OpenCV to use CUDA if available
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("CUDA-enabled OpenCV detected")
        # Use CUDA-accelerated functions where possible
        pass
    
    # Create ArUco dictionary and detector parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # Arrays to store object points and image points from all images
    all_obj_points = []
    all_img_points = []
    all_corners = []
    all_ids = []
    
    # Define the 3D coordinates of the ArUco tag corners
    objp = np.zeros((4, 3), dtype=np.float32)
    objp[:, :2] = np.array([
        [0, 0],
        [tag_size, 0],
        [tag_size, tag_size],
        [0, tag_size]
    ], dtype=np.float32)
    
    # Get list of image files
    image_files = glob.glob(os.path.join(images_path, "*.jpg")) + \
                  glob.glob(os.path.join(images_path, "*.png")) + \
                  glob.glob(os.path.join(images_path, "*.jpeg"))
    
    print(f"Found {len(image_files)} images for calibration")
    
    valid_images = 0
    
    for i, image_file in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_file)}")
        
        # Read image
        image = cv2.imread(image_file)
        if image is None:
            print(f"  Warning: Could not read {image_file}, skipping")
            continue
            
        # Convert to GPU if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(image)
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            gray = gpu_gray.download()
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(gray)
        
        # Check if any markers were detected
        if ids is not None and len(ids) > 0:
            print(f"  Detected {len(ids)} markers")
            
            # Prepare object points and image points for this image
            img_points = []
            obj_points = []
            
            # For each detected marker
            for j, marker_id in enumerate(ids):
                # Add the 4 corner points for this marker
                img_points.extend(corners[j][0])
                
                # Add corresponding 3D points
                obj_points.extend(objp)
            
            # Convert to numpy arrays
            img_points = np.array(img_points, dtype=np.float32)
            obj_points = np.array(obj_points, dtype=np.float32)
            
            # Store for calibration
            all_obj_points.append(obj_points)
            all_img_points.append(img_points)
            all_corners.append(corners)
            all_ids.append(ids)
            
            valid_images += 1
            
            # Optional: Visualize detected markers
            image_with_markers = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
            # cv2.imshow('Detected Markers', image_with_markers)
            # cv2.waitKey(100)
            
        else:
            print(f"  No markers detected in {os.path.basename(image_file)}, skipping")
    
    # cv2.destroyAllWindows()
    
    print(f"\nSuccessfully processed {valid_images} images with markers")
    
    if valid_images < 10:
        print("Warning: Need at least 10 images with detected markers for good calibration!")
        if valid_images == 0:
            print("No valid images found. Check your images and ArUco detection.")
            return None
    
    # Perform camera calibration
    print("Performing camera calibration...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points, all_img_points, gray.shape[::-1], None, None
    )
    
    # Print calibration results
    print(f"\n=== CALIBRATION RESULTS ===")
    print(f"Reprojection error: {ret:.4f}")
    print(f"Camera matrix:")
    print(camera_matrix)
    print(f"Distortion coefficients: {dist_coeffs.ravel()}")
    
    # Calculate and print reprojection errors per image
    print(f"\nReprojection errors per image:")
    mean_errors = []
    for i in range(len(all_obj_points)):
        img_points2, _ = cv2.projectPoints(all_obj_points[i], rvecs[i], tvecs[i], 
                                         camera_matrix, dist_coeffs)
        img_points2 = img_points2.reshape(-1, 2)
        error = cv2.norm(all_img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        mean_errors.append(error)
        print(f"  Image {i+1}: {error:.4f}")
    
    print(f"Mean reprojection error: {np.mean(mean_errors):.4f}")
    
    # Save calibration results
    np.savez(output_file, 
             camera_matrix=camera_matrix, 
             dist_coeffs=dist_coeffs,
             rvecs=rvecs,
             tvecs=tvecs,
             image_size=gray.shape[::-1])
    
    print(f"\nCalibration results saved to {output_file}")
    
    return camera_matrix, dist_coeffs

def estimate_camera_poses(images_path, calibration_file, tag_size=0.06):
    """
    Estimate camera poses for object scan images using ArUco tag and PnP with CUDA acceleration
    """
    # Load camera calibration
    calibration_data = np.load(calibration_file)
    camera_matrix = calibration_data['camera_matrix']
    dist_coeffs = calibration_data['dist_coeffs']
    
    # Create ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # Define 3D coordinates of ArUco tag corners in world space
    object_points = np.array([
        [0, 0, 0],           # bottom-left
        [tag_size, 0, 0],    # bottom-right
        [tag_size, tag_size, 0],  # top-right
        [0, tag_size, 0]     # top-left
    ], dtype=np.float32)
    
    # Get image files
    image_files = sorted(glob.glob(os.path.join(images_path, "*.jpg")) + 
                        glob.glob(os.path.join(images_path, "*.png")) +
                        glob.glob(os.path.join(images_path, "*.jpeg")))
    
    print(f"Found {len(image_files)} images for pose estimation")
    
    camera_data = []
    valid_images = 0
    
    for i, image_file in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_file)}")
        
        # Read image
        image = cv2.imread(image_file)
        if image is None:
            print(f"  Could not read image, skipping")
            continue
            
        H, W = image.shape[:2]
        
        # Use GPU if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(image)
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            gray = gpu_gray.download()
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, _ = detector.detectMarkers(gray)
        
        # Check if at least one tag detected
        if ids is not None and len(ids) >= 1:
            # Reshape corners to match solvePnP input format
            image_points = corners[0].reshape(4, 2).astype(np.float32)
            
            # Estimate camera pose using solvePnP
            success, rvec, tvec = cv2.solvePnP(
                object_points, 
                image_points, 
                camera_matrix, 
                dist_coeffs
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                
                # Create camera-to-world transformation matrix (3x4)
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3] = tvec.reshape(3)
                
                # Invert to get camera-to-world
                c2w = np.linalg.inv(w2c)
                
                camera_data.append({
                    'image_path': image_file,
                    'image': image,
                    'c2w': c2w[:3, :],  # 3x4 camera-to-world matrix
                    'K': camera_matrix,
                    'width': W,
                    'height': H,
                    'rvec': rvec,
                    'tvec': tvec
                })
                
                valid_images += 1
                print(f"  ✓ Pose estimated successfully")
                
                # Visualize detection
                image_with_markers = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
                image_with_axes = cv2.drawFrameAxes(image_with_markers, camera_matrix, dist_coeffs, 
                                                   rvec, tvec, tag_size/2)
                # cv2.imshow('Pose Estimation', image_with_axes)
                # cv2.waitKey(100)
                
            else:
                print(f"  ✗ Pose estimation failed")
        else:
            if ids is None:
                print(f"  ✗ No ArUco tag detected")
            else:
                print(f"  ✗ Found {len(ids)} tags, expected 1")
    
    # cv2.destroyAllWindows()
    print(f"\nSuccessfully estimated poses for {valid_images}/{len(image_files)} images")
    return camera_data

def visualize_camera_frustums(camera_data):
    """
    Visualize camera poses using Viser
    """
    try:
        import importlib
        viser = importlib.import_module('viser')
    except Exception:
        print("viser module not found; attempting to install via pip...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "viser"])
            import importlib
            viser = importlib.import_module('viser')
        except Exception as e:
            print(f"Failed to install or import 'viser': {e}")
            print("Skipping visualization. Install 'viser' manually to enable visualization.")
            return
    
    # Create Viser server
    server = viser.ViserServer(share=True)
    
    print("Starting Viser visualization...")
    print("Check your browser for the visualization interface")
    
    # Add coordinate axes at origin
    server.scene.add_frame(
        "/origin",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
    )

    # Add ArUco tag plane representation
    server.scene.add_box(
        "/aruco_tag",
        dimensions=(0.06, 0.06, 0.001),
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        color=(255, 255, 255),
    )
    
    # Add each camera frustum
    for i, cam in enumerate(camera_data):
        c2w = cam['c2w']
        K = cam['K']
        H = cam['height']
        W = cam['width']
        img = cam['image']
        
        # Calculate field of view
        fov = 2 * np.arctan2(H / 2, K[0, 0])
        aspect = W / H
        
        # Add camera frustum
        server.scene.add_camera_frustum(
            f"/cameras/{i}",
            fov=fov,
            aspect=aspect,
            scale=0.05,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=img
        )
    
    print(f"Added {len(camera_data)} camera frustums to visualization")
    print("Take screenshots for your deliverables from different viewpoints")
    print("The server will keep running until you press Ctrl+C")
    
    # Keep server running
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down visualization server")

def save_camera_poses(camera_data, output_file="camera_poses.json"):
    """Save camera poses to JSON file for later use"""
    poses_dict = {}
    
    for i, cam in enumerate(camera_data):
        poses_dict[os.path.basename(cam['image_path'])] = {
            'c2w': cam['c2w'].tolist(),
            'K': cam['K'].tolist(),
            'image_size': [cam['width'], cam['height']],
            'rvec': cam['rvec'].reshape(-1).tolist(),
            'tvec': cam['tvec'].reshape(-1).tolist()
        }
    
    with open(output_file, 'w') as f:
        json.dump(poses_dict, f, indent=2)
    
    print(f"Camera poses saved to {output_file}")

def create_nerf_dataset(camera_poses_file, calibration_file, images_path,
                        output_file="my_nerf_data.npz",
                        test_size=0.2, val_size=0.1):

    """
    Create NeRF dataset from calibrated images + poses
    Allows interactive choice for handling black boundaries.
    """

    # Load camera poses
    with open(camera_poses_file, 'r') as f:
        camera_poses = json.load(f)

    # Load camera calibration
    calibration_data = np.load(calibration_file)
    camera_matrix = calibration_data['camera_matrix']
    dist_coeffs = calibration_data['dist_coeffs']

    print("Loaded camera matrix:")
    print(camera_matrix)

    images = []
    c2ws = []
    valid_files = []

    # ---- STEP 1: SHOW SAMPLE UNDISTORTED IMAGES ----
    # Pick first 2 images for preview
    preview_imgs = []
    preview_paths = sorted(os.listdir(images_path))[:2]

    print("\nPreviewing undistortion on first 2 images:")

    for name in preview_paths:
        img = cv2.imread(os.path.join(images_path, name))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        und = cv2.undistort(img_rgb, camera_matrix, dist_coeffs)
        preview_imgs.append(und)
        import matplotlib.pyplot as plt
        # Display
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1); plt.imshow(img_rgb); plt.title("Original")
        plt.subplot(1,2,2); plt.imshow(und); plt.title("Undistorted")
        plt.show()

    # Ask user
    use_cropping = input("\nDo you see black boundaries in the undistorted images? (y/n): ").lower().strip() == "y"

    if use_cropping:
        print("\nUsing black-boundary cropping method with updated intrinsics.")
    else:
        print("\nUsing simple undistortion (no cropping).")

    # ---- STEP 2: Determine intrinsics to use ----
    first_img = cv2.imread(os.path.join(images_path, preview_paths[0]))
    h, w = first_img.shape[:2]

    if use_cropping:
        # Compute new intrinsics + ROI
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h),
            alpha=0.1,  # crop aggressively
            newImgSize=(w, h)
        )
        print("\nnew_K (before principal point shift):\n", new_K)
        print("ROI:", roi)

    # ---- STEP 3: PROCESS ENTIRE DATASET ----
    print("\nProcessing all dataset images...")

    for img_name, pose_data in camera_poses.items():
        img_path = os.path.join(images_path, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: {img_name} not found, skipping.")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_name}, skipping.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- UNDISTORT ---
        if use_cropping:
            und = cv2.undistort(img_rgb, camera_matrix, dist_coeffs, None, new_K)

            # Crop ROI
            x, y, w_roi, h_roi = roi
            und = und[y:y+h_roi, x:x+w_roi]

        else:
            und = cv2.undistort(img_rgb, camera_matrix, dist_coeffs)

        # --- CAMERA POSE HANDLING ---
        if 'c2w' in pose_data:
            c2w = np.array(pose_data['c2w'])
        elif 'cam2world' in pose_data:
            c2w = np.array(pose_data['cam2world'])
        else:
            print("Pose missing. Keys:", pose_data.keys())
            continue

        if c2w.shape == (3, 4):
            temp = np.eye(4)
            temp[:3, :] = c2w
            c2w = temp

        images.append(und)
        c2ws.append(c2w)
        valid_files.append(img_name)

    images = np.array(images, dtype=np.uint8)
    c2ws = np.array(c2ws)

    print(f"\nProcessed {len(images)} images.")

    # ---- FINAL FOCAL LENGTH ----
    if use_cropping:
        # Apply principal point offset
        new_K[0,2] -= roi[0]
        new_K[1,2] -= roi[1]
        K_final = new_K
        print("\nAdjusted new_K:\n", K_final)
    else:
        K_final = camera_matrix

    focal = float(K_final[0, 0])
    print(f"\nFinal focal length used = {focal}")

    # ---- DATASET SPLIT ----
    idx = np.arange(len(images))
    train_val_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=180)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size/(1-test_size), random_state=180)

    images_train = images[train_idx]
    c2ws_train = c2ws[train_idx]

    images_val = images[val_idx]
    c2ws_val = c2ws[val_idx]

    c2ws_test = c2ws[test_idx]

    # ---- SAVE ----
    np.savez(
        output_file,
        images_train=images_train,
        c2ws_train=c2ws_train,
        images_val=images_val,
        c2ws_val=c2ws_val,
        c2ws_test=c2ws_test,
        focal=focal
    )

    print(f"\nDataset saved → {output_file}")
    return output_file


def verify_dataset(dataset_file):
    """Verify the created dataset"""
    data = np.load(dataset_file)
    
    print(f"\n=== Dataset Verification ===")
    print(f"images_train: {data['images_train'].shape} (dtype: {data['images_train'].dtype})")
    print(f"c2ws_train: {data['c2ws_train'].shape}")
    print(f"images_val: {data['images_val'].shape}")
    print(f"c2ws_val: {data['c2ws_val'].shape}")
    print(f"c2ws_test: {data['c2ws_test'].shape}")
    print(f"focal: {data['focal']}")
    
    # Check value ranges
    print(f"\nImage value range: [{data['images_train'].min():.1f}, {data['images_train'].max():.1f}]")
    print("✓ Images should be in 0-255 range")
    
    # Check camera poses
    print(f"\nCamera pose norms:")
    print(f"Train: min={np.linalg.norm(data['c2ws_train'][:, :3, 3], axis=1).min():.3f}, "
          f"max={np.linalg.norm(data['c2ws_train'][:, :3, 3], axis=1).max():.3f}")
    
    return data

def load_calibration(calibration_file):
    """Load previously saved calibration data"""
    data = np.load(calibration_file)
    return data['camera_matrix'], data['dist_coeffs']

def main():
    """Main function to run the complete pipeline"""
    print("=== CUDA-Accelerated Camera Calibration and NeRF Dataset Creation ===")
    print(f"Using device: {device}")
    
    # Configuration
    CALIBRATION_IMAGES_PATH = "calibration_tags"  # Folder with calibration images
    OBJECT_IMAGES_PATH = "final_miku_images"           # Folder with object images
    CALIBRATION_FILE = "camera_calibration.npz"
    POSES_FILE = "camera_poses.json"
    NERF_DATASET_FILE = "my_nerf_data.npz"
    TAG_SIZE = 0.06  # 60mm in meters
    
    # Step 1: Camera Calibration
    print("\n" + "="*50)
    print("STEP 1: Camera Calibration")
    print("="*50)
    
    if not os.path.exists(CALIBRATION_FILE):
        print("Calibration file not found. Performing camera calibration...")
        camera_matrix, dist_coeffs = calibrate_camera(CALIBRATION_IMAGES_PATH, TAG_SIZE, CALIBRATION_FILE)
        if camera_matrix is None:
            print("Camera calibration failed! Please check your calibration images.")
            return
    else:
        print("Calibration file found. Loading existing calibration...")
        camera_matrix, dist_coeffs = load_calibration(CALIBRATION_FILE)
        print("Loaded camera matrix:")
        print(camera_matrix)
    
    # Step 2: Camera Pose Estimation
    print("\n" + "="*50)
    print("STEP 2: Camera Pose Estimation")
    print("="*50)
    
    if not os.path.exists(POSES_FILE):
        print("Poses file not found. Estimating camera poses...")
        camera_data = estimate_camera_poses(OBJECT_IMAGES_PATH, CALIBRATION_FILE, TAG_SIZE)
        
        if not camera_data:
            print("Camera pose estimation failed! Please check your object images and ArUco tag placement.")
            return
        
        save_camera_poses(camera_data, POSES_FILE)
    else:
        print("Poses file found. Loading existing poses...")
        with open(POSES_FILE, 'r') as f:
            poses_data = json.load(f)
        print(f"Loaded {len(poses_data)} camera poses")
    
    # Step 3: Viser Visualization (Optional)
    print("\n" + "="*50)
    print("STEP 3: Camera Poses Visualization (Optional)")
    print("="*50)
    
    visualize = input("Do you want to visualize camera poses with Viser? (y/n): ").lower().strip()
    if visualize == 'y':
        if 'camera_data' not in locals():
            # Reload camera data for visualization
            camera_data = estimate_camera_poses(OBJECT_IMAGES_PATH, CALIBRATION_FILE, TAG_SIZE)
        if camera_data:
            visualize_camera_frustums(camera_data)
    
    # Step 4: Create NeRF Dataset
    print("\n" + "="*50)
    print("STEP 4: NeRF Dataset Creation")
    print("="*50)
    
    dataset_path = create_nerf_dataset(
        POSES_FILE,
        CALIBRATION_FILE, 
        OBJECT_IMAGES_PATH,
        output_file=NERF_DATASET_FILE,
        # test_size=0.2,
        test_size = 0.1,
        val_size=0.1
        # val_size = 0.2
    )
    
    if dataset_path:
        # Verify the dataset
        verify_dataset(dataset_path)
        
        print(f"\n✅ Complete Pipeline Finished Successfully!")
        print(f"Your NeRF dataset '{NERF_DATASET_FILE}' is ready for training!")
        print(f"PyTorch version saved as '{NERF_DATASET_FILE.replace('.npz', '.pt')}'")
        print(f"Split information saved to 'dataset_splits.json'")

if __name__ == "__main__":
    main()
