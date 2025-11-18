import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
import cv2
import json

# def create_downsampled_npz(input_path='my_nerf_data.npz', output_path='my_nerf_data_downsampled.npz', target_height=200):
#     """Create a new NPZ file with downsampled images"""
    
#     # Load original data
#     print("Loading original data...")
#     original_data = np.load(input_path)
    
#     print("Original shapes:")
#     print(f"Training images: {original_data['images_train'].shape}")
#     print(f"Validation images: {original_data['images_val'].shape}")
#     print(f"Original image range: {original_data['images_train'].min():.1f} to {original_data['images_train'].max():.1f}")
#     print(f"Original data type: {original_data['images_train'].dtype}")
    
#     def downsample_images(images, target_height):
#         """Downsample images maintaining aspect ratio and [0,255] range"""
#         downsampled = []
#         H, W = images.shape[1:3]
#         new_width = int(W * target_height / H)
        
#         print(f"Downsampling from {H}x{W} to {target_height}x{new_width}")
        
#         for i in range(images.shape[0]):
#             img = images[i]
            
#             # Convert to uint8 if needed
#             if img.dtype != np.uint8:
#                 img = img.astype(np.uint8)
            
#             # Use OpenCV for resizing to preserve exact pixel values
#             img_resized = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)
#             downsampled.append(img_resized)
            
#             if i % 5 == 0:  # Progress indicator
#                 print(f"  Processed {i+1}/{images.shape[0]} images")
        
#         return np.array(downsampled), new_width
    
#     # Downsample training and validation images
#     print("\nDownsampling training images...")
#     images_train_ds, new_width_train = downsample_images(original_data['images_train'], target_height)
    
#     print("\nDownsampling validation images...")
#     images_val_ds, new_width_val = downsample_images(original_data['images_val'], target_height)
    
#     # Adjust focal length for new resolution
#     original_height = original_data['images_train'].shape[1]
#     scale_factor = target_height / original_height
#     adjusted_focal = original_data['focal'] * scale_factor
    
#     # Create new data dictionary
#     new_data = {
#         'images_train': images_train_ds,
#         'c2ws_train': original_data['c2ws_train'],
#         'images_val': images_val_ds, 
#         'c2ws_val': original_data['c2ws_val'],
#         'c2ws_test': original_data['c2ws_test'],
#         'focal': adjusted_focal
#     }
    
#     # Copy K matrix if it exists
#     if 'K' in original_data:
#         # Adjust intrinsic matrix for new resolution
#         K_original = original_data['K']
#         K_new = K_original.copy()
#         K_new[0, 0] *= scale_factor  # fx
#         K_new[1, 1] *= scale_factor  # fy  
#         K_new[0, 2] *= scale_factor  # cx
#         K_new[1, 2] *= scale_factor  # cy
#         new_data['K'] = K_new
#         print(f"Adjusted K matrix for new resolution")
    
#     # Save new NPZ file
#     print(f"\nSaving downsampled data to: {output_path}")
#     np.savez(output_path, **new_data)
    
#     # Verify the new file
#     print("Verifying saved file...")
#     verified_data = np.load(output_path)
    
#     print("\n✅ Downsampled data summary:")
#     print(f"Training images: {verified_data['images_train'].shape}")
#     print(f"Validation images: {verified_data['images_val'].shape}") 
#     print(f"Focal length: {verified_data['focal']:.2f}")
#     print(f"Image range: {verified_data['images_train'].min():.1f} to {verified_data['images_train'].max():.1f}")
#     print(f"Data type: {verified_data['images_train'].dtype}")
    
#     # Show sample images
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
#     # Show original vs downsampled comparison
#     for i in range(3):
#         if i < original_data['images_train'].shape[0]:
#             # Original (resized for display)
#             img_orig = original_data['images_train'][i]
#             if img_orig.dtype != np.uint8:
#                 img_orig = img_orig.astype(np.uint8)
#             img_orig_display = cv2.resize(img_orig, (200, 200), interpolation=cv2.INTER_AREA)
#             img_orig_display = img_orig_display / 255.0  # Only for display
            
#             axes[0, i].imshow(np.clip(img_orig_display, 0, 1))
#             axes[0, i].set_title(f'Original\n{original_data["images_train"].shape[1:3]}')
#             axes[0, i].axis('off')
            
#             # Downsampled (for display)
#             img_ds_display = verified_data['images_train'][i] / 255.0  # Only for display
#             axes[1, i].imshow(np.clip(img_ds_display, 0, 1))
#             axes[1, i].set_title(f'Downsampled\n{verified_data["images_train"].shape[1:3]}')
#             axes[1, i].axis('off')
    
#     plt.tight_layout()
#     plt.savefig('downsampling_comparison.png', dpi=150, bbox_inches='tight')
#     plt.show()
    
#     return output_path

# # Run the function
# if __name__ == '__main__':
#     output_file = create_downsampled_npz(
#         input_path='my_nerf_data.npz',
#         output_path='my_nerf_data_downsampled.npz', 
#         target_height=200
#     )
#     print(f"\n🎉 Downsampled NPZ file created: {output_file}")
#     print("You can now use this file in your training code!")



def create_downsampled_npz(input_path='my_nerf_data.npz', 
                          output_path='my_nerf_data_downsampled.npz', 
                          target_height=200,
                          splits_file='dataset_splits.json'):
    """Create a new NPZ file with downsampled images and adjusted focal length"""
    
    # Load original data
    print("Loading original data...")
    original_data = np.load(input_path)
    
    print("Original shapes:")
    print(f"Training images: {original_data['images_train'].shape}")
    print(f"Validation images: {original_data['images_val'].shape}")
    print(f"Original focal length: {original_data['focal']:.2f}")
    print(f"Original image range: {original_data['images_train'].min():.1f} to {original_data['images_train'].max():.1f}")
    print(f"Original data type: {original_data['images_train'].dtype}")
    
    # Load camera matrix from splits file if available
    camera_matrix = None
    if splits_file and os.path.exists(splits_file):
        with open(splits_file, 'r') as f:
            split_info = json.load(f)
            if 'camera_matrix' in split_info:
                camera_matrix = np.array(split_info['camera_matrix'])
                print(f"\nLoaded camera matrix from {splits_file}:")
                print(camera_matrix)
    
    def downsample_images(images, target_height):
        """Downsample images maintaining aspect ratio and [0,255] range"""
        downsampled = []
        H, W = images.shape[1:3]
        new_width = int(W * target_height / H)
        
        print(f"Downsampling from {H}x{W} to {target_height}x{new_width}")
        
        for i in range(images.shape[0]):
            img = images[i]
            
            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            
            # Use OpenCV INTER_AREA for downsampling (best quality)
            img_resized = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)
            downsampled.append(img_resized)
            
            if (i + 1) % 5 == 0:  # Progress indicator
                print(f"  Processed {i+1}/{images.shape[0]} images")
        
        return np.array(downsampled, dtype=np.uint8), new_width
    
    # Downsample training and validation images
    print("\nDownsampling training images...")
    images_train_ds, new_width = downsample_images(original_data['images_train'], target_height)
    
    print("\nDownsampling validation images...")
    images_val_ds, _ = downsample_images(original_data['images_val'], target_height)
    
    # Calculate scale factor
    original_height = original_data['images_train'].shape[1]
    original_width = original_data['images_train'].shape[2]
    scale_factor = target_height / original_height
    
    print(f"\nScale factor: {scale_factor:.4f}")
    
    # Adjust focal length for new resolution
    adjusted_focal = float(original_data['focal'] * scale_factor)
    
    print(f"Original focal length: {original_data['focal']:.2f}")
    print(f"Adjusted focal length: {adjusted_focal:.2f}")
    
    # Create new data dictionary
    new_data = {
        'images_train': images_train_ds,
        'c2ws_train': original_data['c2ws_train'],
        'images_val': images_val_ds, 
        'c2ws_val': original_data['c2ws_val'],
        'c2ws_test': original_data['c2ws_test'],
        'focal': adjusted_focal
    }
    
    # Adjust camera matrix if available
    if camera_matrix is not None:
        adjusted_camera_matrix = camera_matrix.copy()
        adjusted_camera_matrix[0, 0] *= scale_factor  # fx
        adjusted_camera_matrix[1, 1] *= scale_factor  # fy  
        adjusted_camera_matrix[0, 2] *= scale_factor  # cx
        adjusted_camera_matrix[1, 2] *= scale_factor  # cy
        
        print(f"\nAdjusted camera matrix:")
        print(adjusted_camera_matrix)
        
        # Save updated splits file
        split_info['camera_matrix'] = adjusted_camera_matrix.tolist()
        split_info['focal_length'] = float(adjusted_focal)
        split_info['image_size'] = [int(new_width), int(target_height)]
        
        downsampled_splits_file = splits_file.replace('.json', '_downsampled.json')
        with open(downsampled_splits_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"Saved adjusted split info to {downsampled_splits_file}")
    
    # Save new NPZ file
    print(f"\nSaving downsampled data to: {output_path}")
    np.savez(output_path, **new_data)
    
    # Verify the new file
    print("\nVerifying saved file...")
    verified_data = np.load(output_path)
    
    print("\n✅ Downsampled data summary:")
    print(f"Training images: {verified_data['images_train'].shape}")
    print(f"Validation images: {verified_data['images_val'].shape}") 
    print(f"Test poses: {verified_data['c2ws_test'].shape}")
    print(f"Focal length: {verified_data['focal']:.2f}")
    print(f"Image range: {verified_data['images_train'].min():.1f} to {verified_data['images_train'].max():.1f}")
    print(f"Data type: {verified_data['images_train'].dtype}")
    
    # Show sample images
    print("\nGenerating comparison visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show original vs downsampled comparison
    num_samples = min(3, original_data['images_train'].shape[0])
    for i in range(num_samples):
        # Original (resized for display only)
        img_orig = original_data['images_train'][i]
        if img_orig.dtype != np.uint8:
            img_orig = img_orig.astype(np.uint8)
        # Resize to fixed display size for comparison
        img_orig_display = cv2.resize(img_orig, (300, 300), interpolation=cv2.INTER_AREA)
        img_orig_display = img_orig_display / 255.0  # Normalize for display
        
        axes[0, i].imshow(np.clip(img_orig_display, 0, 1))
        axes[0, i].set_title(f'Original #{i+1}\n{original_data["images_train"].shape[2]}x{original_data["images_train"].shape[1]}')
        axes[0, i].axis('off')
        
        # Downsampled (resized to same display size)
        img_ds = verified_data['images_train'][i]
        img_ds_display = cv2.resize(img_ds, (300, 300), interpolation=cv2.INTER_AREA)
        img_ds_display = img_ds_display / 255.0  # Normalize for display
        
        axes[1, i].imshow(np.clip(img_ds_display, 0, 1))
        axes[1, i].set_title(f'Downsampled #{i+1}\n{verified_data["images_train"].shape[2]}x{verified_data["images_train"].shape[1]}')
        axes[1, i].axis('off')
    
    # Hide extra subplots if we have fewer than 3 images
    for i in range(num_samples, 3):
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Downsampling Comparison (Scale Factor: {scale_factor:.3f})', fontsize=16)
    plt.tight_layout()
    plt.savefig('downsampling_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison image to 'downsampling_comparison.png'")
    plt.close()
    
    return output_path

# Run the function
if __name__ == '__main__':
    import os
    
    output_file = create_downsampled_npz(
        input_path='my_nerf_data.npz',
        output_path='my_nerf_data_downsampled.npz', 
        target_height=240,
        splits_file='dataset_splits.json'
    )
    print(f"\n🎉 Downsampled NPZ file created: {output_file}")
    print("You can now use this file in your training code!")