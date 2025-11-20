"""
DICOM Image Registration System
Integrates ECC and B-spline registration with native DICOM support.
Uses original uint16 data for registration, applies display range only for visualization.

WORKING PRINCIPLE EXPLANATION:

1. DICOM LOADING & PREPROCESSING:
   - Loads original DICOM files with pixel data in uint16 format
   - Extracts 2D frames from DICOM series (handles both single slices and 3D volumes)
   - Preserves original intensity values for accurate registration
   - Display range adjustment only affects visualization, not registration

2. REGISTRATION METHODS:
   
   A. ECC (Enhanced Correlation Coefficient) Registration:
      - Intensity-based alignment using gradient optimization
      - Multiple transformation models:
        * TRANSLATION: Only x,y shifts (2 DOF)
        * EUCLIDEAN: Translation + rotation (3 DOF)  
        * AFFINE: Translation, rotation, scale, shear (6 DOF)
        * HOMOGRAPHY: Full perspective transformation (8 DOF)
      - Works on normalized [0,1] float images but preserves original uint16 data
      - Fast and robust for global alignment

   B. B-spline Non-rigid Registration:
      - Models local deformations using B-spline control points
      - Multi-resolution approach for better convergence
      - Mutual information metric for handling intensity differences
      - Can be applied to full image or specific ROI

   C. Two-stage Registration (ECC + B-spline):
      - Stage 1: ECC for global affine/perspective alignment (ROI or full image, suggest use ROI)
      - Stage 2: B-spline for local non-rigid refinement (ROI or full image, suggest use full)
      - Combines speed of ECC with accuracy of B-spline

3. ROI SUPPORT:
   - Registration can be performed on full image or selected ROI
   - Transformations computed in ROI coordinates are applied to full image
   - Improves speed and focuses on relevant regions

4. PARALLEL PROCESSING:
   - Uses multiprocessing for frame-wise registration
   - Significant speed improvement for large datasets
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging 
import concurrent.futures
from tqdm import tqdm
import time
import SimpleITK as sitk
import glob
import traceback
from multiprocessing import Pool, cpu_count
import pydicom
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("registration_log.txt"),
        logging.StreamHandler()
    ]
)

# ============= DICOM Loading Functions =============

def load_dicom_series(folder_path: str) -> List[Dict]:
    """Load all DICOM files from folder"""
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    logging.info("Scanning DICOM files...")
    
    dicom_files = []
    extensions = ['*.dcm', '*.DCM', '*.dicom', '*.DICOM', '*']
    
    for ext in extensions:
        for file_path in folder.glob(ext):
            if file_path.is_file():
                try:
                    dicom_data = pydicom.dcmread(str(file_path), force=True)
                    
                    if hasattr(dicom_data, 'pixel_array'):
                        image_data = dicom_data.pixel_array
                        
                        file_info = {
                            'file_path': str(file_path),
                            'filename': file_path.name,
                            'data': dicom_data,
                            'image': image_data,
                            'modality': getattr(dicom_data, 'Modality', 'Unknown'),
                            'series_description': getattr(dicom_data, 'SeriesDescription', 'Unknown'),
                            'series_number': getattr(dicom_data, 'SeriesNumber', 0),
                            'instance_number': getattr(dicom_data, 'InstanceNumber', 0),
                            'is_3d': image_data.ndim == 3,
                            'num_frames': image_data.shape[0] if image_data.ndim == 3 else 1,
                            'frame_shape': image_data.shape[1:] if image_data.ndim == 3 else image_data.shape
                        }
                        dicom_files.append(file_info)
                        
                except Exception as e:
                    continue
    
    if not dicom_files:
        raise ValueError("No valid DICOM files found in folder")
    
    logging.info(f"Successfully loaded {len(dicom_files)} DICOM files")
    return dicom_files

def extract_frames_from_series(dicom_files: List[Dict]) -> List[Dict]:
    """Extract 2D frames from DICOM series"""
    all_frames = []
    
    for file_info in dicom_files:
        if not file_info['is_3d']:
            file_info['frame_number'] = file_info['instance_number']
            file_info['total_frames'] = len(dicom_files)
            all_frames.append(file_info)
        else:
            image_3d = file_info['image']
            num_frames = image_3d.shape[0]
            
            for frame_idx in range(num_frames):
                frame_info = file_info.copy()
                frame_info['image'] = image_3d[frame_idx]
                frame_info['frame_number'] = frame_idx + 1
                frame_info['total_frames'] = num_frames
                frame_info['is_3d'] = False
                all_frames.append(frame_info)
    
    all_frames.sort(key=lambda x: x['instance_number'])
    logging.info(f"Extracted {len(all_frames)} frames from DICOM files")
    return all_frames

def adjust_display_range_interactive(image: np.ndarray) -> Tuple[float, float]:
    """Interactively adjust display range"""
    if image.dtype == np.uint8:
        initial_max = 255
    elif image.dtype == np.uint16:
        initial_max = 1500
    else:
        initial_max = np.percentile(image, 95)
    
    vmin, vmax = 0, initial_max
    
    fig, ax = plt.subplots(figsize=(10, 8))
    img_display = ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(f'Display Range: [{vmin:.0f}, {vmax:.0f}]\nPress: y(darker)/n(brighter)/o(ok)')
    
    def on_key(event):
        nonlocal vmin, vmax, img_display, fig
        if event.key in ['y', 'n', 'o']:
            if event.key == 'y':
                vmax = max(10, vmax * 0.8)
                logging.info(f"Decreased range to [{vmin:.0f}, {vmax:.0f}]")
            elif event.key == 'n':
                vmax = vmax * 1.2
                logging.info(f"Increased range to [{vmin:.0f}, {vmax:.0f}]")
            elif event.key == 'o':
                logging.info(f"Final range: [{vmin:.0f}, {vmax:.0f}]")
                plt.close(fig)
                return
            
            img_display.set_clim(vmin, vmax)
            ax.set_title(f'Display Range: [{vmin:.0f}, {vmax:.0f}]\nPress: y(darker)/n(brighter)/o(ok)')
            fig.canvas.draw()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show(block=True)
    
    return (vmin, vmax)

# ============= Preprocessing Functions =============

def normalize_image_for_registration(image: np.ndarray, roi_bounds: Tuple = None) -> np.ndarray:
    """
    Normalize image to [0, 1] for registration
    Input: uint16 original DICOM data
    Output: float32 normalized data
    """
    # Extract ROI if specified
    if roi_bounds:
        x_min, x_max, y_min, y_max = roi_bounds
        roi = image[int(y_min):int(y_max), int(x_min):int(x_max)].copy()
    else:
        roi = image.copy()
    
    # Convert to float32
    roi_float = roi.astype(np.float32)
    
    # Normalize to [0, 1]
    roi_min = roi_float.min()
    roi_max = roi_float.max()
    
    if roi_max > roi_min:
        roi_normalized = (roi_float - roi_min) / (roi_max - roi_min)
    else:
        roi_normalized = np.zeros_like(roi_float)
    
    return roi_normalized

def preprocess_image(img: np.ndarray, use_clahe=False, clahe_params=None, 
                    brightness_delta=0, gaussian_kernel_size=0) -> np.ndarray:
    """
    Preprocess image (CLAHE, brightness, blur)
    Note: Works on normalized [0,1] float32 images
    """
    img_copy = img.copy()
    
    # Apply Gaussian blur if specified
    if gaussian_kernel_size > 0:
        kernel_size = gaussian_kernel_size if gaussian_kernel_size % 2 == 1 else gaussian_kernel_size + 1
        img_copy = cv2.GaussianBlur(img_copy, (kernel_size, kernel_size), 0)
    
    # Adjust brightness (for normalized images)
    if brightness_delta != 0:
        img_copy = np.clip(img_copy + brightness_delta / 255.0, 0, 1)
    
    # Apply CLAHE if enabled (need to convert to uint8 first)
    if use_clahe and clahe_params:
        img_uint8 = (img_copy * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=clahe_params['clip_limit'], 
            tileGridSize=clahe_params['tile_grid_size']
        )
        img_uint8 = clahe.apply(img_uint8)
        img_copy = img_uint8.astype(np.float32) / 255.0
    
    return img_copy

def adjust_preprocessing_parameters(img: np.ndarray) -> Dict:
    """Interactively adjust preprocessing parameters"""
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Initial parameters
    use_clahe = False
    clip_limit = 2.0
    tile_grid_size = (8, 8)
    brightness_delta = 0
    gaussian_kernel_size = 0
    
    while True:
        img_preprocessed = preprocess_image(
            img, 
            use_clahe=use_clahe,
            clahe_params={'clip_limit': clip_limit, 'tile_grid_size': tile_grid_size} if use_clahe else None,
            brightness_delta=brightness_delta,
            gaussian_kernel_size=gaussian_kernel_size
        )
        
        ax.clear()
        ax.imshow(img_preprocessed, cmap='gray', vmin=0, vmax=1)
        title = f"CLAHE: {use_clahe}"
        if use_clahe:
            title += f", clip={clip_limit}, tile={tile_grid_size}"
        title += f"\nbrightness={brightness_delta}, gaussian={gaussian_kernel_size}"
        ax.set_title(title)
        plt.draw()
        plt.pause(0.1)
        
        user_input = input("Adjust parameters? (y/n): ").strip().lower()
        if user_input == 'n':
            break
        elif user_input == 'y':
            try:
                use_clahe = input("Enable CLAHE? (y/n): ").strip().lower() == 'y'
                if use_clahe:
                    clip_limit = float(input("clipLimit (e.g., 2.0): "))
                    tile_grid_size = tuple(map(int, input("tileGridSize (e.g., 8,8): ").split(',')))
                brightness_delta = int(input("Brightness (-255 to 255): "))
                gaussian_kernel_size = int(input("Gaussian kernel (0 or odd number): "))
            except ValueError:
                logging.warning("Invalid input, using current parameters")
    
    plt.ioff()
    plt.close()
    
    return {
        'use_clahe': use_clahe,
        'clip_limit': clip_limit if use_clahe else None,
        'tile_grid_size': tile_grid_size if use_clahe else None,
        'brightness_delta': brightness_delta,
        'gaussian_kernel_size': gaussian_kernel_size
    }

# ============= ECC Registration =============

def select_warp_mode() -> int:
    """
    Interactive selection of ECC transformation mode
    
    Returns:
        int: cv2.MOTION_* constant
    """
    print("\n=== Select ECC Transformation Mode ===")
    print("1. TRANSLATION")
    print("   - Only x, y translation")
    print("   - 2 degrees of freedom")
    print("   - Fastest, most stable")
    print()
    print("2. EUCLIDEAN (Rigid: Translation + Rotation)")
    print("   - Translation + rotation")
    print("   - 3 degrees of freedom")
    print("   - Preserves distances and angles")
    print("   - Recommended for rigid objects")
    print()
    print("3. AFFINE (Translation + Rotation + Scale + Shear)")
    print("   - Full affine transformation")
    print("   - 6 degrees of freedom")
    print("   - Recommended for most cases")
    print()
    print("4. HOMOGRAPHY (Perspective Transformation)")
    print("   - Perspective transformation")
    print("   - 8 degrees of freedom")
    print("   - For planar scenes with perspective distortion")
    print()
    
    while True:
        try:
            choice = input("Select transformation mode (1-4, default=3): ").strip()
            if choice == '':
                choice = '3'
            
            choice_int = int(choice)
            if 1 <= choice_int <= 4:
                break
            else:
                print("Please enter a number between 1 and 4")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    mode_map = {
        1: cv2.MOTION_TRANSLATION,
        2: cv2.MOTION_EUCLIDEAN,
        3: cv2.MOTION_AFFINE,
        4: cv2.MOTION_HOMOGRAPHY
    }
    
    mode_names = {
        cv2.MOTION_TRANSLATION: "TRANSLATION",
        cv2.MOTION_EUCLIDEAN: "EUCLIDEAN (Rigid)",
        cv2.MOTION_AFFINE: "AFFINE",
        cv2.MOTION_HOMOGRAPHY: "HOMOGRAPHY"
    }
    
    selected_mode = mode_map[choice_int]
    logging.info(f"Selected ECC transformation mode: {mode_names[selected_mode]}")
    
    return selected_mode

def get_transformation_info(warp_matrix: np.ndarray, warp_mode: int) -> str:
    """
    Extract transformation information from warp matrix
    
    Args:
        warp_matrix: transformation matrix
        warp_mode: cv2.MOTION_* constant
    
    Returns:
        str: formatted transformation information
    """
    if warp_mode == cv2.MOTION_TRANSLATION:
        tx = warp_matrix[0, 2]
        ty = warp_matrix[1, 2]
        return f"tx={tx:.2f}px, ty={ty:.2f}px"
    
    elif warp_mode == cv2.MOTION_EUCLIDEAN:
        # Extract rotation angle
        angle_rad = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])
        angle_deg = np.degrees(angle_rad)
        tx = warp_matrix[0, 2]
        ty = warp_matrix[1, 2]
        return f"angle={angle_deg:.2f}°, tx={tx:.2f}px, ty={ty:.2f}px"
    
    elif warp_mode == cv2.MOTION_AFFINE:
        # Extract scale
        scale_x = np.sqrt(warp_matrix[0, 0]**2 + warp_matrix[1, 0]**2)
        scale_y = np.sqrt(warp_matrix[0, 1]**2 + warp_matrix[1, 1]**2)
        # Extract rotation
        angle_rad = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])
        angle_deg = np.degrees(angle_rad)
        tx = warp_matrix[0, 2]
        ty = warp_matrix[1, 2]
        return f"angle={angle_deg:.2f}°, scale=({scale_x:.3f},{scale_y:.3f}), tx={tx:.2f}px, ty={ty:.2f}px"
    
    elif warp_mode == cv2.MOTION_HOMOGRAPHY:
        return f"homography matrix (8 DOF)"
    
    return "unknown"

def ecc_register_images(ref_image: np.ndarray, target_image: np.ndarray, 
                        roi_bounds: Tuple = None,
                        warp_mode: int = cv2.MOTION_AFFINE,
                        preprocessing_params: Dict = None) -> Tuple[np.ndarray, float]:
    """
    ECC registration on original uint16 images
    
    Args:
        ref_image: reference image (uint16)
        target_image: target image to align (uint16)
        roi_bounds: ROI for registration (x_min, x_max, y_min, y_max)
        warp_mode: transformation mode (cv2.MOTION_TRANSLATION, EUCLIDEAN, AFFINE, or HOMOGRAPHY)
        preprocessing_params: preprocessing parameters dict
    
    Returns:
        warp_matrix: transformation matrix
        correlation: ECC correlation value
    """
    # Normalize images to [0, 1]
    ref_normalized = normalize_image_for_registration(ref_image, roi_bounds)
    tar_normalized = normalize_image_for_registration(target_image, roi_bounds)
    
    # Apply preprocessing if specified
    if preprocessing_params:
        # Construct correct parameter format
        preprocess_kwargs = {
            'use_clahe': preprocessing_params.get('use_clahe', False),
            'brightness_delta': preprocessing_params.get('brightness_delta', 0),
            'gaussian_kernel_size': preprocessing_params.get('gaussian_kernel_size', 0)
        }
        
        # Only add clahe_params when CLAHE is enabled
        if preprocess_kwargs['use_clahe']:
            preprocess_kwargs['clahe_params'] = {
                'clip_limit': preprocessing_params.get('clip_limit', 2.0),
                'tile_grid_size': preprocessing_params.get('tile_grid_size', (8, 8))
            }
        else:
            preprocess_kwargs['clahe_params'] = None
        
        ref_normalized = preprocess_image(ref_normalized, **preprocess_kwargs)
        tar_normalized = preprocess_image(tar_normalized, **preprocess_kwargs)
    
    # Initialize warp matrix
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # ECC parameters - adjust iterations based on complexity
    if warp_mode == cv2.MOTION_TRANSLATION:
        max_iterations = 5000
    elif warp_mode == cv2.MOTION_EUCLIDEAN:
        max_iterations = 6000
    elif warp_mode == cv2.MOTION_AFFINE:
        max_iterations = 8000
    else:  # HOMOGRAPHY
        max_iterations = 10000
    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, 1e-12)
    
    try:
        correlation, warp_matrix = cv2.findTransformECC(
            ref_normalized, 
            tar_normalized,
            warp_matrix,
            warp_mode,
            criteria,
            inputMask=None,
            gaussFiltSize=5
        )
        return warp_matrix, correlation
    except cv2.error as e:
        logging.error(f"ECC failed: {e}")
        return warp_matrix, 0.0

def apply_ecc_transform_to_full_image(image: np.ndarray, warp_matrix: np.ndarray,
                                     roi_bounds: Tuple = None,
                                     warp_mode: int = cv2.MOTION_AFFINE) -> np.ndarray:
    """
    Apply ECC transform to full image
    Input/Output: uint16 original data
    """
    height, width = image.shape
    
    # Key: ECC matrix needs inversion for correct application
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # For homography matrix
        warp_matrix_inv = np.linalg.inv(warp_matrix)
    else:
        # For affine matrix
        warp_matrix_inv = cv2.invertAffineTransform(warp_matrix)
    
    # Adjust for ROI offset
    if roi_bounds:
        x_min, x_max, y_min, y_max = map(int, roi_bounds)
        adjusted_matrix = warp_matrix_inv.copy()
        
        if warp_mode != cv2.MOTION_HOMOGRAPHY:
            # ROI coordinate system transformation
            # Step 1: Transformation in ROI coordinate system
            # Step 2: Convert back to full image coordinates
            adjusted_matrix[0, 2] = warp_matrix_inv[0, 2] + x_min - warp_matrix_inv[0, 0] * x_min - warp_matrix_inv[0, 1] * y_min
            adjusted_matrix[1, 2] = warp_matrix_inv[1, 2] + y_min - warp_matrix_inv[1, 0] * x_min - warp_matrix_inv[1, 1] * y_min
        else:
            # For homography matrix, more complex coordinate transformation needed
            # Create translation matrices
            T1 = np.eye(3)
            T1[0, 2] = -x_min
            T1[1, 2] = -y_min
            
            T2 = np.eye(3)
            T2[0, 2] = x_min
            T2[1, 2] = y_min
            
            adjusted_matrix = T2 @ warp_matrix_inv @ T1
    else:
        adjusted_matrix = warp_matrix_inv
    
    # Apply transformation
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        transformed = cv2.warpPerspective(
            image, adjusted_matrix, (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
    else:
        transformed = cv2.warpAffine(
            image, adjusted_matrix, (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
    
    return transformed

def ecc_register_single_frame(args):
    """Process single frame with ECC"""
    frame_idx, frame_info, ref_image, roi_bounds, warp_mode, preprocessing_params = args
    
    try:
        target_image = frame_info['image']
        
        # Perform ECC registration
        warp_matrix, correlation = ecc_register_images(
            ref_image, target_image, roi_bounds, warp_mode, preprocessing_params
        )
        
        # Get transformation details for logging
        transform_info = get_transformation_info(warp_matrix, warp_mode)
        
        # Apply transformation to original image
        corrected_image = apply_ecc_transform_to_full_image(
            target_image, warp_matrix, roi_bounds, warp_mode
        )
        
        # Log detailed information
        logging.info(f"Frame {frame_idx}: corr={correlation:.4f}, {transform_info}")
        
        return frame_idx, corrected_image, warp_matrix, correlation
        
    except Exception as e:
        logging.error(f"Error processing frame {frame_idx}: {e}")
        logging.error(traceback.format_exc())
        return frame_idx, frame_info['image'], None, 0.0

# ============= B-spline Registration =============

def bspline_register_single_frame(args):
    """B-spline registration for single frame"""
    frame_idx, moving_image, fixed_image, grid_size, shrink_factors, smoothing_sigmas = args
    
    try:
        # Skip reference frame
        if frame_idx == 0:
            return frame_idx, fixed_image
        
        # Convert numpy arrays to SimpleITK images
        # Normalize to [0, 1] for registration
        fixed_normalized = (fixed_image - fixed_image.min()) / (fixed_image.max() - fixed_image.min())
        moving_normalized = (moving_image - moving_image.min()) / (moving_image.max() - moving_image.min())
        
        fixed_sitk = sitk.GetImageFromArray(fixed_normalized.astype(np.float32))
        moving_sitk = sitk.GetImageFromArray(moving_normalized.astype(np.float32))
        
        # Initialize B-spline transform
        bspline_transform = sitk.BSplineTransformInitializer(fixed_sitk, grid_size, order=3)
        
        # Configure registration
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-6,
            numberOfIterations=50,
            costFunctionConvergenceFactor=1e+7
        )
        registration.SetInitialTransform(bspline_transform)
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        # Execute registration
        final_transform = registration.Execute(fixed_sitk, moving_sitk)
        
        # Apply transformation
        resampled = sitk.Resample(moving_sitk, fixed_sitk, final_transform,
                                 sitk.sitkLinear, 0.0, moving_sitk.GetPixelID())
        
        resampled_array = sitk.GetArrayFromImage(resampled)
        
        # Convert back to original intensity range
        result = resampled_array * (moving_image.max() - moving_image.min()) + moving_image.min()
        result = result.astype(moving_image.dtype)
        
        return frame_idx, result
        
    except Exception as e:
        logging.error(f"B-spline error at frame {frame_idx}: {e}")
        return frame_idx, moving_image

def bspline_register_single_frame_with_roi(args):
    """
    B-spline registration for single frame with ROI support
    
    Args:
        args: tuple containing:
            frame_idx: frame index
            moving_image: moving image (uint16, ECC corrected)
            fixed_image: fixed/reference image (uint16)
            roi_bounds: ROI coordinates (x_min, x_max, y_min, y_max)
            grid_size: B-spline grid size
            shrink_factors: multi-resolution shrink factors
            smoothing_sigmas: multi-resolution smoothing sigmas
    
    Returns:
        tuple: (frame_idx, corrected_image_uint16)
    """
    frame_idx, moving_image, fixed_image, roi_bounds, grid_size, shrink_factors, smoothing_sigmas = args
    
    try:
        # Skip reference frame
        if frame_idx == 0:
            return frame_idx, fixed_image
        
        # Extract ROI if specified
        if roi_bounds:
            x_min, x_max, y_min, y_max = map(int, roi_bounds)
            
            # Ensure ROI bounds are within image dimensions
            h, w = fixed_image.shape
            x_min = max(0, x_min)
            x_max = min(w, x_max)
            y_min = max(0, y_min)
            y_max = min(h, y_max)
            
            # Check if ROI is valid
            if x_max <= x_min or y_max <= y_min:
                logging.warning(f"Invalid ROI for frame {frame_idx}, using full image")
                fixed_roi = fixed_image
                moving_roi = moving_image
                use_roi = False
            else:
                fixed_roi = fixed_image[y_min:y_max, x_min:x_max].copy()
                moving_roi = moving_image[y_min:y_max, x_min:x_max].copy()
                use_roi = True
        else:
            fixed_roi = fixed_image
            moving_roi = moving_image
            use_roi = False
        
        # Store original intensity ranges for restoration
        moving_min = moving_roi.min()
        moving_max = moving_roi.max()
        fixed_min = fixed_roi.min()
        fixed_max = fixed_roi.max()
        
        # Normalize to [0, 1] for registration
        if fixed_max > fixed_min:
            fixed_normalized = (fixed_roi - fixed_min) / (fixed_max - fixed_min)
        else:
            fixed_normalized = np.zeros_like(fixed_roi, dtype=np.float32)
            
        if moving_max > moving_min:
            moving_normalized = (moving_roi - moving_min) / (moving_max - moving_min)
        else:
            moving_normalized = np.zeros_like(moving_roi, dtype=np.float32)
        
        # Convert to SimpleITK images
        fixed_sitk = sitk.GetImageFromArray(fixed_normalized.astype(np.float32))
        moving_sitk = sitk.GetImageFromArray(moving_normalized.astype(np.float32))
        
        # Initialize B-spline transform
        bspline_transform = sitk.BSplineTransformInitializer(fixed_sitk, grid_size, order=3)
        
        # Configure registration
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-6,
            numberOfIterations=50,
            costFunctionConvergenceFactor=1e+7
        )
        registration.SetInitialTransform(bspline_transform)
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        # Execute registration
        final_transform = registration.Execute(fixed_sitk, moving_sitk)
        
        # Apply transformation
        resampled = sitk.Resample(moving_sitk, fixed_sitk, final_transform,
                                 sitk.sitkLinear, 0.0, moving_sitk.GetPixelID())
        
        resampled_array = sitk.GetArrayFromImage(resampled)
        
        # Convert back to original intensity range
        resampled_uint16 = (resampled_array * (moving_max - moving_min) + moving_min).astype(moving_roi.dtype)
        
        # Apply ROI result to full image
        if use_roi:
            # Create result image by copying original and updating ROI
            result_image = moving_image.copy()
            result_image[y_min:y_max, x_min:x_max] = resampled_uint16
        else:
            result_image = resampled_uint16
        
        logging.info(f"B-spline completed for frame {frame_idx} {'with ROI' if use_roi else 'full image'}")
        return frame_idx, result_image
        
    except Exception as e:
        logging.error(f"B-spline error at frame {frame_idx}: {e}")
        logging.error(traceback.format_exc())
        return frame_idx, moving_image

# ============= Main Processing Pipeline =============
def select_frame_range(all_frames: List[Dict]) -> List[Dict]:
    """
    Interactive frame range selection
    
    Args:
        all_frames: List of all frames
    
    Returns:
        Selected frames
    """
    total_frames = len(all_frames)
    
    print("\n" + "="*60)
    print("     Frame Range Selection")
    print("="*60)
    print(f"Total frames available: {total_frames}")
    print()
    print("Options:")
    print("1. Process all frames")
    print("2. Select frame range (e.g., frame 10 to 50)")
    print("3. Use first N frames (quick test)")
    print()
    
    while True:
        try:
            choice = input("Select option (1-3, default=1): ").strip()
            if choice == '' or choice == '1':
                # Process all frames
                logging.info(f"Processing all {total_frames} frames")
                return all_frames
            
            elif choice == '2':
                # Select range
                print(f"\nFrame indices range from 1 to {total_frames}")
                
                while True:
                    try:
                        start_input = input(f"Enter start frame (1-{total_frames}, default=1): ").strip()
                        start_frame = int(start_input) if start_input else 1
                        
                        end_input = input(f"Enter end frame ({start_frame}-{total_frames}, default={total_frames}): ").strip()
                        end_frame = int(end_input) if end_input else total_frames
                        
                        # Validate range
                        if start_frame < 1 or start_frame > total_frames:
                            print(f"Error: Start frame must be between 1 and {total_frames}")
                            continue
                        
                        if end_frame < start_frame or end_frame > total_frames:
                            print(f"Error: End frame must be between {start_frame} and {total_frames}")
                            continue
                        
                        # Convert to 0-based indices
                        start_idx = start_frame - 1
                        end_idx = end_frame
                        
                        selected_frames = all_frames[start_idx:end_idx]
                        num_selected = len(selected_frames)
                        
                        logging.info(f"Selected frames {start_frame} to {end_frame} ({num_selected} frames)")
                        print(f"✓ Selected {num_selected} frames (frame {start_frame} to {end_frame})")
                        
                        return selected_frames
                    
                    except ValueError:
                        print("Invalid input. Please enter valid frame numbers.")
            
            elif choice == '3':
                # First N frames
                while True:
                    try:
                        n_input = input(f"Enter number of frames to process (1-{total_frames}, default=100): ").strip()
                        n_frames = int(n_input) if n_input else min(100, total_frames)
                        
                        if n_frames < 1 or n_frames > total_frames:
                            print(f"Error: Number must be between 1 and {total_frames}")
                            continue
                        
                        selected_frames = all_frames[:n_frames]
                        logging.info(f"Processing first {n_frames} frames")
                        print(f"✓ Selected first {n_frames} frames")
                        
                        return selected_frames
                    
                    except ValueError:
                        print("Invalid input. Please enter a valid number.")
            
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        
        except KeyboardInterrupt:
            print("\nOperation cancelled. Using all frames.")
            return all_frames

def process_dicom_with_registration(folder_path: str, method: str = 'ecc',
                                    roi_bounds: Tuple = None,
                                    display_range: Tuple = (0, 500),
                                    max_workers: int = None,
                                    warp_mode: int = None):
    """
    Main processing pipeline for DICOM registration
    
    Args:
        folder_path: DICOM folder path
        method: 'ecc', 'bspline', or 'two-stage'
        roi_bounds: ROI for registration (x_min, x_max, y_min, y_max)
        display_range: Display window (vmin, vmax) for visualization
        max_workers: Number of parallel workers
        warp_mode: ECC transformation mode (only for 'ecc' and 'two-stage')
    """
    logging.info(f"=" * 60)
    logging.info(f"Starting {method.upper()} Registration")
    logging.info(f"=" * 60)
    
    # Load DICOM files
    dicom_files = load_dicom_series(folder_path)
    all_frames_full = extract_frames_from_series(dicom_files)
    
    # ============= New: Frame range selection =============
    all_frames = select_frame_range(all_frames_full)
    
    # Ensure reference frame is in selected range
    if len(all_frames) == 0:
        raise ValueError("No frames selected!")
    
    # Get reference image (always use the first frame in selected range)
    reference_index = 0
    ref_image = all_frames[reference_index]['image']
    
    logging.info(f"Total frames to process: {len(all_frames)}")
    logging.info(f"Image dtype: {ref_image.dtype}")
    logging.info(f"Image range: [{ref_image.min()}, {ref_image.max()}]")
    logging.info(f"Image shape: {ref_image.shape}")
    
    # Adjust display range interactively
    if display_range is None:
        display_range = adjust_display_range_interactive(ref_image)
    
    # ============= Registration Region Selection =============
    if method != 'two-stage':
        # Single-stage methods: ECC or B-spline only
        print("\n=== Registration Region Selection ===")
        print("1. Full image registration")
        print("   - Register the entire image")
        print("   - Slower but considers all image features")
        print()
        print("2. ROI-based registration")
        print("   - Register only within selected ROI")
        print("   - Faster, focuses on region of interest")
        print("   - Transformation computed in ROI, applied to full image")
        print()
        
        region_choice = input("Select registration region (1-2, default=2): ").strip()
        
        if region_choice == '1':
            # Full image registration
            use_roi_for_registration = False
            registration_roi = None
            logging.info("Using FULL IMAGE for registration")
            print("✓ Full image registration selected")
        else:
            # ROI registration (default)
            use_roi_for_registration = True
            if roi_bounds is None:
                roi_bounds = select_roi_interactive(ref_image, display_range)
            registration_roi = roi_bounds
            logging.info(f"Using ROI for registration: {roi_bounds}")
            print(f"✓ ROI-based registration selected: ({roi_bounds[0]:.0f}, {roi_bounds[1]:.0f}, {roi_bounds[2]:.0f}, {roi_bounds[3]:.0f})")
    
    # Select warp mode for ECC-based methods
    if method in ['ecc', 'two-stage'] and warp_mode is None:
        warp_mode = select_warp_mode()
    elif warp_mode is None:
        warp_mode = cv2.MOTION_AFFINE  # Default for other methods
    
    # Ask about preprocessing
    use_preprocessing = input("\nUse preprocessing (CLAHE/brightness/blur)? (y/n): ").strip().lower() == 'y'
    
    if use_preprocessing:
        # For single-stage methods, use registration_roi if defined
        # For two-stage, will be handled separately
        if method != 'two-stage':
            ref_normalized = normalize_image_for_registration(ref_image, registration_roi)
        else:
            ref_normalized = normalize_image_for_registration(ref_image, None)
        preprocessing_params = adjust_preprocessing_parameters(ref_normalized)
    else:
        preprocessing_params = None
    
    # Set number of workers
    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)
    
    logging.info(f"Using {max_workers} parallel workers")
    
    # Perform registration based on method
    corrected_frames = {}
    
    if method == 'ecc':
        logging.info("Performing ECC registration...")
        
        # Prepare arguments
        args_list = [
            (i, frame_info, ref_image, registration_roi, warp_mode, preprocessing_params)
            for i, frame_info in enumerate(all_frames) if i != reference_index
        ]
        
        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(ecc_register_single_frame, args) for args in args_list]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="ECC Registration"):
                frame_idx, corrected, warp_matrix, corr = future.result()
                corrected_frames[frame_idx] = corrected
        
        # Add reference frame
        corrected_frames[reference_index] = ref_image
        
    elif method == 'bspline':
        logging.info("Performing B-spline registration...")
        
        grid_size = [4, 4]
        shrink_factors = [4, 2, 1]
        smoothing_sigmas = [2, 1, 0]
        
        if use_roi_for_registration:
            # Use ROI-based B-spline registration
            logging.info("Using ROI-based B-spline registration")
            args_list = [
                (i, frame_info['image'], ref_image, registration_roi, grid_size, shrink_factors, smoothing_sigmas)
                for i, frame_info in enumerate(all_frames)
            ]
            
            with Pool(processes=max_workers) as pool:
                results = []
                for result in tqdm(pool.imap(bspline_register_single_frame_with_roi, args_list), 
                                 total=len(args_list), desc="B-spline Registration (ROI)"):
                    results.append(result)
        else:
            # Full image B-spline registration
            logging.info("Using full image B-spline registration")
            args_list = [
                (i, frame_info['image'], ref_image, grid_size, shrink_factors, smoothing_sigmas)
                for i, frame_info in enumerate(all_frames)
            ]
            
            with Pool(processes=max_workers) as pool:
                results = []
                for result in tqdm(pool.imap(bspline_register_single_frame, args_list), 
                                 total=len(args_list), desc="B-spline Registration (Full)"):
                    results.append(result)
        
        for frame_idx, corrected in results:
            corrected_frames[frame_idx] = corrected
            
    elif method == 'two-stage':
        logging.info("=" * 60)
        logging.info("Two-Stage Registration: ECC + B-spline")
        logging.info("=" * 60)
        
        # ============= Two-stage independent region selection =============
        print("\n=== Two-Stage Registration Region Selection ===")
        print("Configure registration regions for each stage separately:")
        print()
        
        # Stage 1 (ECC) region selection
        print("--- Stage 1: ECC Registration ---")
        print("1. Full image")
        print("2. ROI-based (faster, focused)")
        print()
        ecc_region_choice = input("Select ECC registration region (1-2, default=2): ").strip()
        
        if ecc_region_choice == '1':
            use_roi_for_ecc = False
            ecc_roi_bounds = None
            logging.info("Stage 1 (ECC): Using FULL IMAGE")
            print("✓ Stage 1 (ECC): Full image selected")
        else:
            use_roi_for_ecc = True
            if roi_bounds is None:
                # If ROI not selected before, select now
                print("\nSelect ROI for ECC registration:")
                ecc_roi_bounds = select_roi_interactive(ref_image, display_range)
            else:
                # Use previously selected ROI
                ecc_roi_bounds = roi_bounds
            logging.info(f"Stage 1 (ECC): Using ROI {ecc_roi_bounds}")
            print(f"✓ Stage 1 (ECC): ROI selected ({ecc_roi_bounds[0]:.0f}, {ecc_roi_bounds[1]:.0f}, {ecc_roi_bounds[2]:.0f}, {ecc_roi_bounds[3]:.0f})")
        
        # Stage 2 (B-spline) region selection
        print("\n--- Stage 2: B-spline Registration ---")
        print("1. Full image (refine entire image)")
        print("2. Same ROI as Stage 1 (consistent refinement)")
        print("3. Different ROI (select new ROI for B-spline)")
        print()
        
        bspline_region_choice = input("Select B-spline registration region (1-3, default=2): ").strip()
        
        if bspline_region_choice == '1':
            use_roi_for_bspline = False
            bspline_roi_bounds = None
            logging.info("Stage 2 (B-spline): Using FULL IMAGE")
            print("✓ Stage 2 (B-spline): Full image selected")
        elif bspline_region_choice == '3':
            use_roi_for_bspline = True
            print("\nSelect NEW ROI for B-spline registration:")
            bspline_roi_bounds = select_roi_interactive(ref_image, display_range)
            logging.info(f"Stage 2 (B-spline): Using NEW ROI {bspline_roi_bounds}")
            print(f"✓ Stage 2 (B-spline): New ROI selected ({bspline_roi_bounds[0]:.0f}, {bspline_roi_bounds[1]:.0f}, {bspline_roi_bounds[2]:.0f}, {bspline_roi_bounds[3]:.0f})")
        else:  # default: option 2
            if use_roi_for_ecc:
                use_roi_for_bspline = True
                bspline_roi_bounds = ecc_roi_bounds
                logging.info(f"Stage 2 (B-spline): Using SAME ROI as Stage 1: {bspline_roi_bounds}")
                print(f"✓ Stage 2 (B-spline): Same ROI as Stage 1")
            else:
                use_roi_for_bspline = False
                bspline_roi_bounds = None
                logging.info("Stage 2 (B-spline): Using FULL IMAGE (Stage 1 used full image)")
                print("✓ Stage 2 (B-spline): Full image (Stage 1 used full image)")
        
        print()
        logging.info("Two-stage configuration complete")
        
        # Stage 1: ECC Registration
        mode_names = {
            cv2.MOTION_TRANSLATION: "TRANSLATION",
            cv2.MOTION_EUCLIDEAN: "EUCLIDEAN (Rigid)",
            cv2.MOTION_AFFINE: "AFFINE",
            cv2.MOTION_HOMOGRAPHY: "HOMOGRAPHY"
        }
        ecc_region_desc = "ROI" if use_roi_for_ecc else "Full Image"
        logging.info(f"Stage 1: ECC Registration ({mode_names[warp_mode]}) on {ecc_region_desc}...")
        
        ecc_corrected_frames = {}
        
        # Prepare arguments for ECC
        args_list = [
            (i, frame_info, ref_image, ecc_roi_bounds, warp_mode, preprocessing_params)
            for i, frame_info in enumerate(all_frames) if i != reference_index
        ]
        
        # Process ECC in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(ecc_register_single_frame, args) for args in args_list]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Stage 1: ECC"):
                frame_idx, corrected, warp_matrix, corr = future.result()
                ecc_corrected_frames[frame_idx] = corrected
        
        # Add reference frame
        ecc_corrected_frames[reference_index] = ref_image
        
        logging.info(f"Stage 1 completed: {len(ecc_corrected_frames)} frames aligned with ECC")
        
        # Stage 2: B-spline Registration on ECC results
        bspline_region_desc = "ROI" if use_roi_for_bspline else "Full Image"
        logging.info(f"Stage 2: B-spline Registration (Non-rigid Refinement) on {bspline_region_desc}...")
        
        grid_size = [4, 4]
        shrink_factors = [4, 2, 1]
        smoothing_sigmas = [2, 1, 0]
        
        if use_roi_for_bspline:
            # Use ROI-based B-spline
            args_list = [
                (i, ecc_corrected_frames[i], ref_image, bspline_roi_bounds, grid_size, shrink_factors, smoothing_sigmas)
                for i in sorted(ecc_corrected_frames.keys())
            ]
            
            with Pool(processes=max_workers) as pool:
                results = []
                for result in tqdm(pool.imap(bspline_register_single_frame_with_roi, args_list), 
                                 total=len(args_list), desc="Stage 2: B-spline (ROI)"):
                    results.append(result)
        else:
            # Full image B-spline
            args_list = [
                (i, ecc_corrected_frames[i], ref_image, grid_size, shrink_factors, smoothing_sigmas)
                for i in sorted(ecc_corrected_frames.keys())
            ]
            
            with Pool(processes=max_workers) as pool:
                results = []
                for result in tqdm(pool.imap(bspline_register_single_frame, args_list), 
                                 total=len(args_list), desc="Stage 2: B-spline (Full)"):
                    results.append(result)
        
        for frame_idx, corrected in results:
            corrected_frames[frame_idx] = corrected
        
        logging.info(f"Stage 2 completed: {len(corrected_frames)} frames refined with B-spline")
        logging.info("Two-stage registration completed!")
    
    # Save results
    save_corrected_frames(all_frames, corrected_frames, display_range, method)
    
    return all_frames, corrected_frames, display_range

def select_roi_interactive(image: np.ndarray, display_range: Tuple) -> Tuple:
    """Interactive ROI selection"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image, cmap='gray', vmin=display_range[0], vmax=display_range[1])
    ax.set_title('Click two diagonal corners for ROI')
    
    plt.tight_layout()
    points = plt.ginput(2, timeout=0, show_clicks=True)
    plt.close(fig)
    
    if len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        logging.info(f"ROI selected: ({x_min:.0f}, {y_min:.0f}) to ({x_max:.0f}, {y_max:.0f})")
        return (x_min, x_max, y_min, y_max)
    else:
        # Default ROI
        h, w = image.shape
        return (w*0.25, w*0.75, h*0.25, h*0.75)

def save_corrected_frames(all_frames: List[Dict], corrected_frames: Dict[int, np.ndarray],
                         display_range: Tuple, method: str):
    """Save corrected frames and original frames for comparison"""
    
    # Create output folders
    output_folder_corrected = f"corrected_{method}"
    output_folder_original = f"original_{method}"
    os.makedirs(output_folder_corrected, exist_ok=True)
    os.makedirs(output_folder_original, exist_ok=True)
    
    logging.info(f"Saving original frames to {output_folder_original}...")
    logging.info(f"Saving corrected frames to {output_folder_corrected}...")
    
    for i in sorted(corrected_frames.keys()):
        # Get original image
        original_image = all_frames[i]['image']
        corrected_image = corrected_frames[i]
        
        # Apply display range and convert to uint8 (original image)
        original_display = np.clip(original_image, display_range[0], display_range[1])
        original_display = ((original_display - display_range[0]) / 
                           (display_range[1] - display_range[0]) * 255).astype(np.uint8)
        
        # Apply display range and convert to uint8 (corrected image)
        corrected_display = np.clip(corrected_image, display_range[0], display_range[1])
        corrected_display = ((corrected_display - display_range[0]) / 
                            (display_range[1] - display_range[0]) * 255).astype(np.uint8)
        
        # Save original image
        original_path = os.path.join(output_folder_original, f"original_{i+1:03d}.tif")
        cv2.imwrite(original_path, original_display)
        
        # Save corrected image
        corrected_path = os.path.join(output_folder_corrected, f"corrected_{i+1:03d}.tif")
        cv2.imwrite(corrected_path, corrected_display)
    
    logging.info(f"Saved {len(corrected_frames)} original frames and {len(corrected_frames)} corrected frames")

def main():
    folder_path = "./52beb3c2/214be1a6/6c1a8228"
    
    if not os.path.exists(folder_path):
        logging.error(f"Folder not found: {folder_path}")
        return
    
    print("\n" + "="*60)
    print("     DICOM Image Registration System")
    print("="*60)
    print("\nAvailable Registration Methods:")
    print("-" * 60)
    print("1. ECC (Enhanced Correlation Coefficient)")
    print("   - Fast rigid/affine transformation")
    print("   - Multiple transformation modes available:")
    print("     * Translation (2 DOF)")
    print("     * Euclidean/Rigid (3 DOF)")
    print("     * Affine (6 DOF)")
    print("     * Homography (8 DOF)")
    print()
    print("2. B-spline")
    print("   - Non-rigid deformation")
    print("   - Captures local deformations")
    print("   - Slower but more flexible")
    print()
    print("3. Two-stage (ECC + B-spline)")
    print("   - Hybrid approach")
    print("   - Stage 1: ECC for global alignment")
    print("   - Stage 2: B-spline for local refinement")
    print("   - Best quality, longest processing time")
    print("-" * 60)
    
    choice = input("\nSelect method (1-3): ").strip()
    
    method_map = {'1': 'ecc', '2': 'bspline', '3': 'two-stage'}
    method = method_map.get(choice, 'ecc')
    
    try:
        all_frames, corrected_frames, display_range = process_dicom_with_registration(
            folder_path,
            method=method,
            roi_bounds=None,
            display_range=None,
            max_workers=None,
            warp_mode=None
        )
        
        print("\n" + "="*60)
        print("     Registration Completed Successfully!")
        print("="*60)
        print(f"\nTotal frames processed: {len(corrected_frames)}")
        print(f"Original frames folder: original_{method}/")
        print(f"Corrected frames folder: corrected_{method}/")
        print(f"Display range used: [{display_range[0]:.0f}, {display_range[1]:.0f}]")
        print("\nCheck registration_log.txt for detailed information.")
        print("="*60)
        
        logging.info("Registration completed successfully!")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        traceback.print_exc()
        print("\n" + "="*60)
        print("     Registration Failed!")
        print("="*60)
        print(f"Error: {e}")
        print("Check registration_log.txt for details.")
        print("="*60)

if __name__ == "__main__":
    main()