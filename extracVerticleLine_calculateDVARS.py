import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def bresenham_line(x0, y0, x1, y1):
    """Generate all integer coordinate points on a line (Bresenham's algorithm)"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

def input_line_segment(image_path, output_folder):
    """Get line segment coordinates and corresponding Y coordinates"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image {image_path} could not be read")
    height, width = img.shape

    # Display the image and input coordinates
    plt.imshow(img, cmap='gray')
    plt.title("Input the two endpoints of the line segment")
    plt.axis('off')
    plt.show()

    print(f"Please select a line segment in the image {image_path} (image dimensions: width {width}, height {height})")
    x1, y1 = map(int, input("Enter the coordinates of the first point (x1, y1): ").split(","))
    x2, y2 = map(int, input("Enter the coordinates of the second point (x2, y2): ").split(","))

    # Check if coordinates are valid
    if not (0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height):
        print("Error: Coordinates are out of image bounds")
        return None, None

    # Generate line points and extract Y coordinates
    points = bresenham_line(x1, y1, x2, y2)
    y_coords = [y for (x, y) in points]

    # Print the number of pixels on the line
    print(f"Number of pixels on the line: {len(points)}")

    # Draw and save the result
    img_with_line = cv2.line(img.copy(), (x1, y1), (x2, y2), 255, 2)
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate a unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_folder, f"output_with_line_{timestamp}.png")
    
    # Save the image
    success = cv2.imwrite(output_path, img_with_line)
    if success:
        print(f"Image saved: {output_path}")
    else:
        print(f"Failed to save image: {output_path}")
    
    plt.imshow(img_with_line, cmap='gray')
    plt.title("Input Line Segment")
    plt.axis('off')
    plt.show()

    return points, y_coords

def extract_pixel_intensity(image_path, points):
    """Extract intensity values of all points on the line"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image {image_path} could not be read")
    return [img[y, x] if (0 <= x < img.shape[1] and 0 <= y < img.shape[0]) else 0 for (x, y) in points]

def calculate_dvars(intensity_data):
    """
    Calculate DVARS (Derivative of VARiance over time)
    
    Parameters:
        intensity_data: Intensity data with shape (T, n), where T is the number of frames, n is the number of pixels
        
    Returns:
        dvars: DVARS value
    """
    T, n = intensity_data.shape
    print(f"T={T}")  # Print the number of frames
    print(f"n={n}")  # Print the number of pixels
    rms_num = []  # Store RMS values for each frame

    # Convert intensity_data to float to avoid overflow
    intensity_data = intensity_data.astype(np.float32)

    for t in range(1, T):  # Start from the second frame
        total_sum = 0  # Reset the sum
        for i in range(n):
            # Use absolute difference
            diff = intensity_data[t][i] - intensity_data[t-1][i]  # Calculate the difference between adjacent frames

            total_sum += float(diff ** 2)  # Accumulate the squared differences
            #print(f"t={t}, i={i}, diff={diff}, total_sum={total_sum}")  # Print debug information

        rms = np.sqrt(total_sum / n)  # Calculate the RMS for the current frame
        rms_num.append(rms)  # Add the RMS to the list
        #print(f"Frame {t}: RMS={rms}")  # Print the RMS for the current frame

    dvars = np.mean(rms_num)  # Calculate the mean of all RMS values
    #print(f"Length of num: {len(rms_num)}")
    return dvars

def process_image_series(image_folder, output_folder):
    """Process the image series and return intensity data and Y coordinates"""
    image_files = sorted(os.listdir(image_folder))
    if not image_files:
        print("No images found in the folder")
        return None, None

    # Select the line segment in the first frame
    first_image = os.path.join(image_folder, image_files[0])
    points, y_coords = input_line_segment(first_image, output_folder)
    if not points:
        return None, None

    # Extract intensity values from all images
    intensity_data = []
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        intensities = extract_pixel_intensity(img_path, points)
        intensity_data.append(intensities)
    
    # Convert intensity data to a NumPy array
    intensity_data = np.array(intensity_data)

    # Check the type and shape of intensity_data
    # print(f"Type of intensity_data: {intensity_data.dtype}")
    # print(f"Shape of intensity_data: {intensity_data.shape}")

    # Calculate DVARS
    dvars = calculate_dvars(intensity_data)
    print(f"DVARS: {dvars:.4f}")
    
    return intensity_data, y_coords

def plot_pixel_intensity(data, y_coords, output_folder):
    """Plot a heatmap: X-axis is time, Y-axis is Y coordinates, color represents intensity"""
    if data is None or y_coords is None or data.size == 0 or len(y_coords) == 0:
        print("No valid data")
        return
    
    intensity_matrix = np.array(data)
    
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(intensity_matrix.T, cmap='gray', shading='auto')
    plt.colorbar(label="Pixel Intensity")
    
    # Set axis labels
    plt.xlabel("Time (Frame Index)")
    plt.ylabel("Y Coordinate")
    
    # Dynamically adjust Y-axis ticks
    n_ticks = min(20, len(y_coords))  # Display at most 20 ticks
    step = max(1, len(y_coords) // n_ticks)
    plt.yticks(
        np.arange(0, len(y_coords), step),  # Tick positions
        y_coords[::step],  # Tick labels (actual Y coordinates)
        rotation=45,
        fontsize=8
    )
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate a unique filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_folder, f"pixel_intensity_heatmap_{timestamp}.png")
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Heatmap saved as: {output_path}")
    plt.show()

# Example usage
#image_folder = '/hpc/tpal287/stomach_segmentation/motion_removed_tiff_files/case10_images'
image_folder = './Bspline_output_method2_pre'  #Bsplline_output_parallel ECC_preprocess_parallel ECC_output_parallel
#image_folder = '/hpc/lsu876/MRI_seg/python_code/motionCorrection/case10/withPreprocess_output'
output_folder = './output_images' 
intensity_data, y_coords = process_image_series(image_folder, output_folder)
if intensity_data is not None:
    plot_pixel_intensity(intensity_data, y_coords, output_folder)
