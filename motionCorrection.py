"""
A comprehensive image registration system integrating multiple registration methods, 
including ECC affine transformation, B-spline non-rigid deformation, and two-stage registration.
The system provides interactive parameter adjustment, parallel processing, and complete logging capabilities.

Key Features:
------------
1. **Multiple Registration Methods**:
   - ECC (Enhanced Correlation Coefficient) - Fast rigid transformation
   - B-spline - Precise non-rigid deformation
   - Two-stage registration - Hybrid approach (ECC followed by B-spline)

2. **Flexible Preprocessing**:
   - CLAHE contrast enhancement (with interactive parameter adjustment)
   - Brightness adjustment
   - Gaussian blur denoising
   - Optional preprocessing pipeline

3. **Efficient Parallel Processing**:
   - Multi-threading/multi-processing parallel execution

Usage Instructions:
------------------
1. Set input image folder path
2. Select number of parallel workers
3. Choose registration method (1-3)
4. Adjust preprocessing parameters as needed

Output Structure:
----------------
- ECC registration results: ./ECC_output/
- B-spline registration results: ./Bspline_output/
- Preprocessed images: ./preprocessed_images/
- Log file: registration_log.txt

Technical Stack:
---------------
- OpenCV (ECC registration)
- SimpleITK (B-spline registration)
- Matplotlib (interactive parameter adjustment)
- Concurrent.futures (parallel processing)
- TQDM (progress bars)

Parameters Configuration:
------------------------
ECC Registration:
- warp_mode: cv2.MOTION_AFFINE (default) or cv2.MOTION_HOMOGRAPHY
- iterations: 8000
- termination_epsilon: 1e-12

B-spline Registration:
- grid_size: [4, 4]
- shrink_factors: [4, 2, 1]
- smoothing_sigmas: [2, 1, 0]
- iterations: 50
- histogram_bins: 50

Preprocessing:
- CLAHE: clipLimit=1.5, tileGridSize=(16,16)
- Brightness: adjustable from -255 to 255
- Gaussian kernel: adjustable odd sizes (3,5,7,etc.)
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("registration_log.txt"),
        logging.StreamHandler()
    ]
)

def preprocess_image(img, use_clahe=True, clahe_params=None, brightness_delta=0, gaussian_kernel_size=0):
    """Image preprocessing function with optional CLAHE."""
    img_copy = img.copy()
    
    # Apply Gaussian blur if specified
    if gaussian_kernel_size > 0:
        # Ensure kernel size is odd
        kernel_size = gaussian_kernel_size if gaussian_kernel_size % 2 == 1 else gaussian_kernel_size + 1
        img_copy = cv2.GaussianBlur(img_copy, (kernel_size, kernel_size), 0)
    
    # Adjust brightness
    img_copy = cv2.add(img_copy, brightness_delta)
    
    # Apply CLAHE if enabled
    if use_clahe and clahe_params:
        clahe = cv2.createCLAHE(clipLimit=clahe_params['clip_limit'], 
                               tileGridSize=clahe_params['tile_grid_size'])
        img_copy = clahe.apply(img_copy)
    
    return img_copy

def adjust_preprocessing(img):
    """Adjust preprocessing parameters and display the image."""
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Initial parameters
    use_clahe = True
    clip_limit = 1.5
    tile_grid_size = (16, 16)
    brightness_delta = 0      # 0 means no brightness adjustment
    gaussian_kernel_size = 0  # 0 means no Gaussian filter
    
    while True:
        # Create a copy of the image for processing
        img_preprocessed = preprocess_image(
            img, 
            use_clahe=use_clahe,
            clahe_params={'clip_limit': clip_limit, 'tile_grid_size': tile_grid_size},
            brightness_delta=brightness_delta,
            gaussian_kernel_size=gaussian_kernel_size
        )
        
        # Display the image
        ax.clear()
        ax.imshow(img_preprocessed, cmap='gray')
        title = f"CLAHE Enabled: {use_clahe}\n"
        if use_clahe:
            title += f"clipLimit={clip_limit}, tileGridSize={tile_grid_size}\n"
        title += f"brightness={brightness_delta}, gaussian_kernel={gaussian_kernel_size}"
        ax.set_title(title)
        plt.draw()
        plt.pause(0.1)
        
        # User input
        user_input = input("Adjust parameters? (y/n): ").strip().lower()
        logging.info(f"User input for adjusting parameters: {user_input}")
        if user_input == 'n':
            break
        elif user_input == 'y':
            try:
                use_clahe_input = input("Enable CLAHE? (y/n): ").strip().lower()
                use_clahe = use_clahe_input == 'y'
                
                if use_clahe:
                    clip_limit = float(input("Enter new clipLimit (e.g., 1.5): "))
                    tile_grid_size = tuple(map(int, input("Enter new tileGridSize (e.g., 16,16): ").split(',')))
                
                brightness_delta = int(input("Enter brightness adjustment (-255 to 255, 0 for none): "))
                gaussian_kernel_size = int(input("Enter Gaussian kernel size (0 for none, or odd number like 3,5,7): "))
                
                log_msg = f"Parameters - CLAHE enabled: {use_clahe}"
                if use_clahe:
                    log_msg += f", clipLimit: {clip_limit}, tileGridSize: {tile_grid_size}"
                log_msg += f", brightness: {brightness_delta}, gaussian_kernel: {gaussian_kernel_size}"
                logging.info(log_msg)
            except ValueError:
                logging.warning("Invalid input! Using default parameters.")
                use_clahe = True
                clip_limit = 1.5
                tile_grid_size = (16, 16)
                brightness_delta = 0
                gaussian_kernel_size = 0
        else:
            logging.warning("Invalid input! Please enter 'y' or 'n'.")
    
    plt.ioff()
    plt.close()
    
    preprocessing_params = {
        'use_clahe': use_clahe,
        'clip_limit': clip_limit if use_clahe else None,
        'tile_grid_size': tile_grid_size if use_clahe else None,
        'brightness_delta': brightness_delta,
        'gaussian_kernel_size': gaussian_kernel_size
    }
    
    return preprocessing_params

def mse(image1, image2):
    """Compute the mean squared error between two images"""
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

def align_images(img_ref, img_target, warp_mode=cv2.MOTION_AFFINE):
    """Align the target image to the reference image using ECC"""
    # Initialize the warp matrix
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Set the termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 8000, 1e-12)

    # Run ECC to find the warp matrix
    try:
        _, warp_matrix = cv2.findTransformECC(img_ref, img_target, warp_matrix, warp_mode, criteria)
    except Exception as e:
        logging.error(f"Error during alignment: {e}")  
        return None

    # Warp the target image
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        img_aligned = cv2.warpPerspective(img_target, warp_matrix, (img_ref.shape[1], img_ref.shape[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        img_aligned = cv2.warpAffine(img_target, warp_matrix, (img_ref.shape[1], img_ref.shape[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return img_aligned

def preprocess_single_image(args):
    """Preprocess a single image"""
    image_file, image_folder, preprocess_output_folder, preprocessing_params, reference_size = args
    
    try:
        # Load the target image
        target_image_path = os.path.join(image_folder, image_file)
        img_target = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
        if img_target is None:
            logging.error(f"Failed to load target image: {target_image_path}")  
            return image_file, None
        
        # Ensure the target image has the same size as the reference image
        img_target = cv2.resize(img_target, reference_size)
        
        # Apply preprocessing to the target image
        img_target_preprocessed = preprocess_image(
            img_target,
            use_clahe=preprocessing_params['use_clahe'],
            clahe_params={
                'clip_limit': preprocessing_params['clip_limit'],
                'tile_grid_size': preprocessing_params['tile_grid_size']
            } if preprocessing_params['use_clahe'] else None,
            brightness_delta=preprocessing_params['brightness_delta'],
            gaussian_kernel_size=preprocessing_params['gaussian_kernel_size']
        )
        
        # Save the preprocessed target image
        preprocessed_target_image_path = os.path.join(preprocess_output_folder, f"preprocessed_{image_file}")
        cv2.imwrite(preprocessed_target_image_path, img_target_preprocessed)
        
        return image_file, preprocessed_target_image_path
    except Exception as e:
        logging.error(f"Error preprocessing image {image_file}: {str(e)}")
        return image_file, None

def align_single_image(args):
    """Align a single image"""
    image_file, img_ref_path, img_target_path, output_folder, warp_mode = args
    
    try:
        # Load the reference and target images
        img_ref = cv2.imread(img_ref_path, cv2.IMREAD_GRAYSCALE)
        img_target = cv2.imread(img_target_path, cv2.IMREAD_GRAYSCALE)
        
        if img_ref is None or img_target is None:
            logging.error(f"Failed to load images for alignment: {image_file}")
            return image_file, None, None
        
        # Align the target image to the reference image
        img_aligned = align_images(img_ref, img_target, warp_mode)
        if img_aligned is None:
            logging.error(f"Alignment failed for image: {image_file}")  
            return image_file, None, None
        
        # Save the aligned image
        aligned_output_path = os.path.join(output_folder, f"ECC_{image_file}")
        cv2.imwrite(aligned_output_path, img_aligned)
        
        # Calculate MSE
        error = mse(img_ref, img_aligned)
        
        return image_file, aligned_output_path, error
    except Exception as e:
        logging.error(f"Error aligning image {image_file}: {str(e)}")
        return image_file, None, None

def process_batch(function, args_list, max_workers, desc):
    """Process a batch of items with progress bar"""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(function, args): args for args in args_list}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(args_list), desc=desc):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                args = futures[future]
                logging.error(f"Exception during {desc} for {args[0]}: {str(e)}")
    
    return results

def preprocess_image_sequence(image_folder, preprocess_output_folder, method_name="", reference_index=0, max_workers=None):
    """General preprocessing function for ECC and B-spline registration"""
    # Load all images from the folder
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    if not image_files:
        logging.error("No images found in the folder!")  
        return None

    # Load the reference image
    ref_image_path = os.path.join(image_folder, image_files[reference_index])
    img_ref = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)
    if img_ref is None:
        logging.error(f"Failed to load reference image: {ref_image_path}")  
        return None

    # Ask the user if preprocessing is required
    method_text = f" for {method_name}" if method_name else ""
    preprocess_choice = input(f"Do you want to preprocess the images{method_text}? (y/n): ").strip().lower()
    logging.info(f"User choice for preprocessing{method_text}: {preprocess_choice}") 
    
    if preprocess_choice != 'y':
        logging.info(f"Skipping preprocessing{method_text}, using original images.")
        return {
            'input_folder': image_folder,
            'preprocessing_params': None,
            'reference_size': (img_ref.shape[1], img_ref.shape[0])
        }
    
    # Set number of parallel workers
    if max_workers is None:
        max_workers = os.cpu_count()
    
    # Create output folder
    os.makedirs(preprocess_output_folder, exist_ok=True)
    
    # Store reference image dimensions
    reference_size = (img_ref.shape[1], img_ref.shape[0])
    
    # Adjust preprocessing parameters
    preprocessing_params = adjust_preprocessing(img_ref)
    
    logging.info(f"Final {method_name} preprocessing parameters: CLAHE enabled: {preprocessing_params['use_clahe']}, " +
                (f"clipLimit={preprocessing_params['clip_limit']}, " +
                 f"tileGridSize={preprocessing_params['tile_grid_size']}, " if preprocessing_params['use_clahe'] else "") +
                f"brightness={preprocessing_params['brightness_delta']}, " +
                f"gaussian_kernel={preprocessing_params['gaussian_kernel_size']}")

    # Apply final preprocessing parameters to the reference image
    img_ref_preprocessed = preprocess_image(
        img_ref,
        use_clahe=preprocessing_params['use_clahe'],
        clahe_params={
            'clip_limit': preprocessing_params['clip_limit'],
            'tile_grid_size': preprocessing_params['tile_grid_size']
        } if preprocessing_params['use_clahe'] else None,
        brightness_delta=preprocessing_params['brightness_delta'],
        gaussian_kernel_size=preprocessing_params['gaussian_kernel_size']
    )

    # Display the final preprocessing result
    window_title = f'Preprocessed Reference Image{method_text}'
    cv2.imshow(window_title, img_ref_preprocessed)
    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC key to exit
            break
        if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

    # Save the preprocessed reference image
    preprocessed_ref_image_path = os.path.join(preprocess_output_folder, f"preprocessed_{image_files[reference_index]}")
    cv2.imwrite(preprocessed_ref_image_path, img_ref_preprocessed)
    logging.info(f"Saved preprocessed reference image{method_text}: {preprocessed_ref_image_path}")
    
    # Preprocess all images in parallel
    logging.info(f"Preprocessing all images{method_text}...")
    
    # Process images in smaller batches to avoid memory issues
    batch_size = min(50, len(image_files))
    num_batches = (len(image_files) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]
        
        logging.info(f"Processing preprocessing batch {batch_idx + 1}/{num_batches} ({len(batch_files)} images)")
        
        # Prepare arguments for parallel preprocessing
        preprocess_args_list = [
            (image_file, image_folder, preprocess_output_folder, preprocessing_params, reference_size)
            for image_file in batch_files
        ]
        
        # Process batch
        batch_results = process_batch(
            preprocess_single_image, 
            preprocess_args_list, 
            max_workers, 
            f"Preprocessing batch {batch_idx + 1}/{num_batches}"
        )
        
        # Log preprocessed image paths
        for image_file, preprocessed_path in batch_results:
            if preprocessed_path is not None:
                logging.info(f"Saved preprocessed image{method_text}: {preprocessed_path}")
        
        # Add a small delay between batches
        time.sleep(0.5)
    
    logging.info(f"{method_name} preprocessing completed!")
    return {
        'input_folder': preprocess_output_folder,
        'preprocessing_params': preprocessing_params,
        'reference_size': reference_size
    }

def ecc_align_image_sequence(image_folder, preprocess_output_folder, output_folder, reference_index=0, warp_mode=cv2.MOTION_AFFINE, max_workers=None):
    """Align a sequence of images to a reference image with parallel processing using ECC"""
    # Preprocess images if needed
    preprocess_result = preprocess_image_sequence(
        image_folder, 
        preprocess_output_folder, 
        method_name="ECC",
        reference_index=reference_index,
        max_workers=max_workers
    )
    
    if preprocess_result is None:
        logging.error("Failed to preprocess images for ECC registration!")
        return
    
    input_folder_for_ecc = preprocess_result['input_folder']
    preprocessing_params = preprocess_result['preprocessing_params']
    reference_size = preprocess_result['reference_size']
    
    # Load all images from the input folder
    image_files = sorted([f for f in os.listdir(input_folder_for_ecc) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    if not image_files:
        logging.error("No images found in the folder!")  
        return

    # Load the reference image
    ref_image_path = os.path.join(input_folder_for_ecc, image_files[reference_index])
    img_ref = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)
    if img_ref is None:
        logging.error(f"Failed to load reference image: {ref_image_path}")  
        return

    # Set number of parallel workers
    if max_workers is None:
        max_workers = os.cpu_count()
    logging.info(f"Using {max_workers} parallel workers for ECC alignment")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the reference image
    ref_output_path = os.path.join(output_folder, f"ECC_{image_files[reference_index]}")
    cv2.imwrite(ref_output_path, img_ref)
    logging.info(f"Saved reference image: {ref_output_path}")
    
    # Create a list of image files to process (excluding the reference image)
    image_files_to_process = [f for f in image_files if image_files.index(f) != reference_index]
    logging.info(f"Processing {len(image_files_to_process)} images for ECC alignment")
    
    # Process alignment in batches
    batch_size = min(50, len(image_files_to_process))
    num_batches = (len(image_files_to_process) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_files_to_process))
        batch_files = image_files_to_process[start_idx:end_idx]
        
        logging.info(f"Processing ECC alignment batch {batch_idx + 1}/{num_batches} ({len(batch_files)} images)")
        
        # Prepare arguments for parallel alignment
        align_args_list = []
        for image_file in batch_files:
            target_image_path = os.path.join(input_folder_for_ecc, image_file)
            align_args_list.append((
                image_file, 
                ref_image_path, 
                target_image_path, 
                output_folder, 
                warp_mode
            ))
        
        # Process batch
        batch_results = process_batch(
            align_single_image, 
            align_args_list, 
            max_workers, 
            f"ECC aligning batch {batch_idx + 1}/{num_batches}"
        )
        
        # Log alignment results
        for image_file, aligned_path, error in batch_results:
            if aligned_path is not None and error is not None:
                logging.info(f"Saved aligned image: {aligned_path}")
                logging.info(f"Mean Squared Error for {image_file}: {error}")
        
        # Add a small delay between batches to allow resources to be released
        time.sleep(0.5)
    
    return output_folder

def bspline_register_image(args):
    """Register a single image using B-spline registration"""
    i, moving_image_path, fixed_image, output_path, grid_size, shrink_factors, smoothing_sigmas = args
    
    try:
        # Skip the first frame (fixed image)
        if i == 0:
            fixed_array = sitk.GetArrayFromImage(fixed_image)
    
            # 归一化到 [0, 255]
            if np.ptp(fixed_array) > 0:
                rescaled = (fixed_array - fixed_array.min()) / np.ptp(fixed_array) * 255
            else:
                rescaled = np.zeros_like(fixed_array)
            
            # 转换为 uint8
            final_image = sitk.GetImageFromArray(rescaled.astype(np.uint8))
            final_image.CopyInformation(fixed_image)
            
            sitk.WriteImage(final_image, output_path)
            return f"Frame {i}: Fixed image saved"
        
        # Read moving image
        moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
        
        # Initialize B-spline transform
        bspline_transform = sitk.BSplineTransformInitializer(fixed_image, grid_size, order=3)
        
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
        
        # Set multi-resolution parameters
        registration.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        # Execute registration
        final_transform = registration.Execute(fixed_image, moving_image)
        
        # Apply transformation
        resampled_image = sitk.Resample(moving_image, fixed_image, final_transform,
                                       sitk.sitkLinear, 0.0, moving_image.GetPixelID())
        
        # Normalize and convert to uint8
        resampled_array = sitk.GetArrayFromImage(resampled_image)
        if np.ptp(resampled_array) > 0:
            rescaled = (resampled_array - resampled_array.min()) / np.ptp(resampled_array) * 255
        else:
            rescaled = np.zeros_like(resampled_array)
        
        final_image = sitk.GetImageFromArray(rescaled.astype(np.uint8))
        final_image.CopyInformation(resampled_image)
        
        # Save result
        sitk.WriteImage(final_image, output_path)
        return f"Frame {i}: Registration completed"
    
    except Exception as e:
        error_msg = f"Frame {i}: Error - {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


def bspline_align_image_sequence(input_folder, output_folder, grid_size=[4, 4], 
                                 shrink_factors=[4, 2, 1], smoothing_sigmas=[2, 1, 0], 
                                 max_workers=None):
    """Align a sequence of images using B-spline registration"""
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load images - support multiple extensions and case-insensitive
    image_files = []
    all_files = os.listdir(input_folder)
    image_files = sorted([os.path.join(input_folder, f) for f in all_files 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    
    # Fallback: try glob patterns
    if not image_files:
        for pattern in ['*.tif', '*.tiff', '*.TIF', '*.TIFF', '*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
            files = sorted(glob.glob(os.path.join(input_folder, pattern)))
            if files:
                image_files = files
                logging.info(f"Found {len(files)} images with pattern {pattern}")
                break
    
    if not image_files:
        # Display detailed error information
        logging.error(f"No image files found in folder: {input_folder}")
        logging.error(f"Total files in folder: {len(all_files)}")
        logging.error(f"First 10 files: {all_files[:10]}")
        raise ValueError(f"No .tif/.tiff/.png/.jpg/.jpeg files found in: {input_folder}")
    
    logging.info(f"Found {len(image_files)} images for B-spline registration")
    
    # Pre-load fixed image
    fixed_image = sitk.ReadImage(image_files[0], sitk.sitkFloat32)
    
    # Set number of parallel workers
    if max_workers is None:
        max_workers = max(1, int(cpu_count() * 0.4))
    
    logging.info(f"Starting B-spline registration with {max_workers} workers...")
    
    # Prepare arguments list (pass pre-loaded fixed_image object)
    args_list = []
    for i, moving_path in enumerate(image_files):
        output_path = os.path.join(output_folder, f"Bspline_{i+1:03d}.tif")
        args_list.append((i, moving_path, fixed_image, output_path, 
                         grid_size, shrink_factors, smoothing_sigmas))
    
    # Execute parallel processing
    with Pool(processes=max_workers) as pool:
        results = []
        for idx, result in enumerate(pool.imap(bspline_register_image, args_list)):
            print(f"Processing progress: {idx+1}/{len(image_files)}")
            results.append(result)
    
    # Log results
    for res in results:
        logging.info(res)
    
    logging.info("B-spline registration completed!")
    return output_folder

def get_registration_choice():
    """Get user choice for registration method"""
    print("\n=== Registration Method Selection ===")
    print("1. ECC (MOTION_AFFINE) - Fast, rigid transformation")
    print("2. B-spline - Slow, non-rigid deformation")
    print("3. Two-stage (ECC + B-spline) - Rigid followed by non-rigid")
    
    while True:
        try:
            choice = int(input("Select registration method (1-3): ").strip())
            if 1 <= choice <= 3:
                return choice
            else:
                print("Please enter a number between 1 and 3")
        except ValueError:
            print("Please enter a valid number")

def main():
    # Set input and output folders
    image_folder = "./images_from_MRI"  
    
    # Get number of CPU cores for parallel processing
    max_cores = os.cpu_count()
    print(f"Available CPU cores: {max_cores}")
    
    # Allow user to set the number of parallel workers
    try:
        num_workers = int(input(f"Enter number of parallel workers (1-{max_cores}, default={max(1, max_cores-1)}): ").strip())
        if num_workers < 1 or num_workers > max_cores:
            num_workers = max(1, max_cores-1)
            print(f"Using default: {num_workers} workers")
    except ValueError:
        num_workers = max(1, max_cores-1)
        print(f"Using default: {num_workers} workers")
    
    # Get registration method choice
    reg_choice = get_registration_choice()
    
    # Execute selected registration method
    if reg_choice == 1:
        # ECC registration only
        preprocess_output_folder = "./ECC_preprocess"
        ecc_output_folder = "./ECC_output_method1_pre"
        
        logging.info("Starting ECC registration...")
        ecc_align_image_sequence(
            image_folder, 
            preprocess_output_folder, 
            ecc_output_folder, 
            reference_index=0, 
            warp_mode=cv2.MOTION_AFFINE,
            max_workers=num_workers
        )
        logging.info("ECC registration completed!")
        
    elif reg_choice == 2:
        # B-spline registration only
        preprocess_output_folder = "./Bspline_preprocess"
        bspline_output_folder = "./Bspline_output_method2_pre"
        
        logging.info("Starting B-spline registration...")
        
        # Preprocess images for B-spline if needed
        preprocess_result = preprocess_image_sequence(
            image_folder,
            preprocess_output_folder,
            method_name="B-spline",
            reference_index=0,
            max_workers=num_workers
        )
        
        if preprocess_result is None:
            logging.error("Failed to preprocess images for B-spline registration!")
            return
        
        input_folder_for_bspline = preprocess_result['input_folder']
        
        # Perform B-spline registration
        bspline_align_image_sequence(
            input_folder_for_bspline,
            bspline_output_folder,
            grid_size=[4, 4],
            shrink_factors=[4, 2, 1],
            smoothing_sigmas=[2, 1, 0],
            max_workers=num_workers
        )
        logging.info("B-spline registration completed!")
        
    elif reg_choice == 3:
        # Two-stage registration: ECC followed by B-spline
        preprocess_output_folder = "./preprocessed_images"
        ecc_output_folder = "./ECC_output"
        bspline_output_folder = "./Bspline_output"
        
        # First stage: ECC registration
        logging.info("Starting first stage: ECC registration...")
        ecc_result_folder = ecc_align_image_sequence(
            image_folder, 
            preprocess_output_folder, 
            ecc_output_folder, 
            reference_index=0, 
            warp_mode=cv2.MOTION_AFFINE,
            max_workers=num_workers
        )
        logging.info("ECC registration completed!")
        
        # Second stage: B-spline registration on ECC results
        logging.info("Starting second stage: B-spline registration...")
        bspline_align_image_sequence(
            ecc_result_folder,
            bspline_output_folder,
            grid_size=[4, 4],
            shrink_factors=[4, 2, 1],
            smoothing_sigmas=[2, 1, 0],
            max_workers=num_workers
        )
        logging.info("Two-stage registration completed!")

if __name__ == "__main__":
    main()