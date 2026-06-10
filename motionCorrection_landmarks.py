import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2
from scipy.optimize import minimize
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import warnings
import json
warnings.filterwarnings('ignore')

class Weasis3DMotionCorrection:
    def __init__(self):
        self.reference_image = None
        self.reference_landmarks = None
        self.image_size = None
        self.roi_bounds = None
        self.display_range = (0, 500)
        self.current_frame = 0
        self.data_storage = {}  
        self.case_number = None
        self.series_number = None
        self.first_in_series = None
        self.last_landmark_image = None
        
    def load_dicom_series(self, folder_path: str) -> List[Dict]:
        """Load all DICOM files in folder, handle 3D data"""
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        print("Scanning DICOM files...")
        
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
                                'study_description': getattr(dicom_data, 'StudyDescription', 'Unknown'),
                                'patient_name': getattr(dicom_data, 'PatientName', 'Unknown'),
                                'is_3d': image_data.ndim == 3,
                                'num_frames': image_data.shape[0] if image_data.ndim == 3 else 1,
                                'frame_shape': image_data.shape[1:] if image_data.ndim == 3 else image_data.shape
                            }
                            dicom_files.append(file_info)
                            print(f"✓ Loaded: {file_path.name} - Shape: {image_data.shape}")
                            
                    except Exception as e:
                        continue
        
        if not dicom_files:
            raise ValueError("No valid DICOM files found in folder")
        
        print(f"Successfully loaded {len(dicom_files)} DICOM files")
        return dicom_files
    
    def analyze_series(self, dicom_files: List[Dict]):
        """Analyze series information in DICOM files"""
        series_dict = {}
        
        for file_info in dicom_files:
            series_key = (
                file_info['series_number'], 
                file_info['series_description'],
                file_info['modality']
            )
            
            if series_key not in series_dict:
                series_dict[series_key] = {
                    'series_number': file_info['series_number'],
                    'series_description': file_info['series_description'],
                    'modality': file_info['modality'],
                    'files': [],
                    'instance_numbers': set(),
                    'patient_name': file_info['patient_name'],
                    'study_description': file_info['study_description'],
                    'is_3d_series': file_info['is_3d'],
                    'is_2d_sequence': not file_info['is_3d'],  
                    'total_frames': 0,
                    'total_files': 0
                }
            
            series_dict[series_key]['files'].append(file_info)
            series_dict[series_key]['instance_numbers'].add(file_info['instance_number'])
            series_dict[series_key]['total_files'] += 1
            
            if file_info['is_3d']:
                series_dict[series_key]['total_frames'] += file_info['num_frames']
            else:
                series_dict[series_key]['total_frames'] += 1 
        
        return series_dict
    
    def select_series_interactive(self, series_dict: Dict) -> List[Dict]:
        """Interactively select series to process"""
        print("\n" + "=" * 60)
        print("Detected DICOM Series:")
        print("=" * 60)
        
        series_list = list(series_dict.keys())
        
        for i, (series_num, series_desc, modality) in enumerate(series_list):
            series_info = series_dict[(series_num, series_desc, modality)]
            
            if series_info['is_3d_series']:
                type_info = f"3D Series - {series_info['total_frames']} frames"
            elif series_info['is_2d_sequence']:
                type_info = f"2D Sequence - {series_info['total_files']} images"
            else:
                type_info = f"Series - {series_info['total_files']} files"
            
            print(f"{i+1}. [{modality}] Series {series_num}: {series_desc}")
            print(f"    {type_info}")
            print(f"    Patient: {series_info['patient_name']}, Study: {series_info['study_description']}")
            print()
        
        while True:
            try:
                choice = input(f"Select series to process (1-{len(series_list)}, or press Enter for first): ").strip()
                
                if not choice:
                    selected_key = series_list[0]
                    break
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(series_list):
                    selected_key = series_list[choice_idx]
                    break
                else:
                    print("Invalid selection, please try again")
                    
            except ValueError:
                print("Please enter a valid number")
        
        selected_series = series_dict[selected_key]
        selected_files = selected_series['files']
        
        selected_files.sort(key=lambda x: x['instance_number'])
        
        print(f"\nSelected: [{selected_series['modality']}] Series {selected_series['series_number']} - {selected_series['series_description']}")
        if selected_series['is_3d_series']:
            print(f"Contains {len(selected_files)} 3D files, total {selected_series['total_frames']} frames")
        else:
            print(f"Contains {len(selected_files)} 2D images")
        
        return selected_files
    
    def extract_frames_from_3d(self, dicom_files: List[Dict]) -> List[Dict]:
        """process 2D DICOM series"""
        all_frames = []
        
        for file_info in dicom_files:
            if not file_info['is_3d']:
                file_info['frame_number'] = file_info['instance_number']
                file_info['total_frames'] = len(dicom_files)
                file_info['is_2d_series'] = True
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
                    frame_info['filename'] = f"{file_info['filename']}_frame{frame_idx+1:03d}"
                    all_frames.append(frame_info)
        
        all_frames.sort(key=lambda x: x['instance_number'])
        
        print(f"Processed {len(all_frames)} 2D frames from {len(dicom_files)} DICOM files")
        return all_frames
    
    def adjust_display_range(self, image: np.ndarray) -> float:
        """Adjust display range for single frame - COMPLETELY FIXED VERSION"""
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape: {image.shape}")
        
        if image.dtype == np.uint8:
            mmm = 255
        elif image.dtype == np.uint16:
            mmm = 1500
        else:
            mmm = np.percentile(image, 95)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        img_display = ax.imshow(image, cmap='gray', vmin=0, vmax=mmm)
        ax.set_title(f'Display Range: [0, {mmm:.1f}]\nPress: y(darker)/n(brighter)/o(ok)')
        
        def on_key(event):
            nonlocal mmm, img_display, ax, fig
            if event.key in ['y', 'n', 'o']:
                if event.key == 'y':  
                    mmm = max(10, mmm * 0.8)
                    print(f"Decreased range to [0, {mmm:.1f}]")
                elif event.key == 'n':  
                    mmm = mmm * 1.2
                    print(f"Increased range to [0, {mmm:.1f}]")
                elif event.key == 'o':  
                    print(f"Final range: [0, {mmm:.1f}]")
                    plt.close(fig)
                    return
                
                img_display.set_clim(0, mmm)
                ax.set_title(f'Display Range: [0, {mmm:.1f}]\nPress: y(darker)/n(brighter)/o(ok)')
                fig.canvas.draw()

        fig.canvas.mpl_connect('key_press_event', on_key)
        
        print("Adjust display range:")
        print("  Press 'y': Image too dark - decrease range")
        print("  Press 'n': Image too bright - increase range") 
        print("  Press 'o': OK - accept current range")
        print("→ Now focus on the image window and press y/n/o")
        
        plt.tight_layout()
        plt.show(block=True)  
        
        self.display_range = (0, mmm)
        return mmm

    def select_roi(self, image: np.ndarray) -> Tuple[float, float, float, float]:
        """Select region of interest - FIXED VERSION"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image, cmap='gray', vmin=self.display_range[0], vmax=self.display_range[1])
        ax.set_title('Click two points to define ROI (diagonal corners of rectangle)')
        
        height, width = image.shape
        ax.text(0.02, 0.98, f'Image size: {width} x {height}', 
                transform=ax.transAxes, color='white', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        
        print("Click two points to define ROI...")
        print("First point: top-left corner")
        print("Second point: bottom-right corner")
        
        points = plt.ginput(2, timeout=0, show_clicks=True)
        plt.close(fig)
        
        if len(points) == 2:
            point1, point2 = points
            x1, y1 = point1
            x2, y2 = point2
            
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)
            
            roi_bounds = (x_min, x_max, y_min, y_max)
            
            roi_width = x_max - x_min
            roi_height = y_max - y_min
            print(f"✓ ROI selected successfully:")
            print(f"  Top-left: ({x_min:.1f}, {y_min:.1f})")
            print(f"  Bottom-right: ({x_max:.1f}, {y_max:.1f})")
            print(f"  ROI size: {roi_width:.1f} x {roi_height:.1f} pixels")
            
            self._visualize_roi_selection(image, roi_bounds)
            
            return roi_bounds
        else:
            height, width = image.shape
            default_bounds = (width*0.25, width*0.75, height*0.25, height*0.75)
            print(f"Using default ROI: {default_bounds}")
            return default_bounds

    def _visualize_roi_selection(self, image: np.ndarray, roi_bounds: Tuple[float, float, float, float]):
        x_min, x_max, y_min, y_max = roi_bounds
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.imshow(image, cmap='gray', vmin=self.display_range[0], vmax=self.display_range[1])
        rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                            linewidth=3, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        ax1.set_title('Full Image with ROI (Red Rectangle)')
        
        roi_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        ax2.imshow(roi_image, cmap='gray', vmin=self.display_range[0], vmax=self.display_range[1])
        ax2.set_title(f'ROI Region Only\nSize: {roi_image.shape[1]} x {roi_image.shape[0]}')
        
        plt.tight_layout()
        plt.suptitle('ROI Selection Confirmation - Close to continue', fontsize=14)
        plt.show()
        
        response = input("Is this ROI correct? (y/n): ").strip().lower()
        plt.close()
        
        if response == 'n':
            print("Let's select ROI again...")
            return self.select_roi(image)
        else:
            print("ROI confirmed!")
            return roi_bounds

    def sum_sqrd_error(self, params: np.ndarray, ref_pts: np.ndarray, tar_pts: np.ndarray) -> float:
        """Calculate transformation error"""
        dx, dy, theta = params
        theta_rad = np.deg2rad(theta)
        
        R = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)]
        ])
        
        transformed_pts = (R @ tar_pts.T).T + np.array([dx, dy])
        error = np.sum((transformed_pts - ref_pts) ** 2)
        return error
    
    def calculate_transformation(self, ref_landmarks: np.ndarray, tar_landmarks: np.ndarray) -> np.ndarray:
        """Calculate motion correction parameters"""
        image_shape = self.image_size
        center = np.array([image_shape[1]/2, image_shape[0]/2])
        
        ref_pts_centered = ref_landmarks - center
        tar_pts_centered = tar_landmarks - center
        
        initial_guess = [0, 0, 0]
        result = minimize(self.sum_sqrd_error, initial_guess,
                         args=(ref_pts_centered, tar_pts_centered),
                         method='Nelder-Mead',
                         options={'maxiter': 1000})
        
        if result.success:
            return result.x
        else:
            raise RuntimeError(f"Motion correction optimization failed: {result.message}")
    
    def apply_correction(self, image: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Apply motion correction"""
        dx, dy, theta = params
        height, width = image.shape
        
        M_rotate = cv2.getRotationMatrix2D((width/2, height/2), theta, 1.0)
        rotated = cv2.warpAffine(image, M_rotate, (width, height), 
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,  borderValue=0) # ← 填充值为0（黑色）
        
        M_translate = np.float32([[1, 0, dx], [0, 1, dy]])
        corrected = cv2.warpAffine(rotated, M_translate, (width, height),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,  borderValue=0)
        
        return corrected

    def save_landmarks_to_file(self, output_file: str = "landmarks_data.json"):
        landmarks_data = {}
        
        if self.case_number in self.data_storage:
            case_data = self.data_storage[self.case_number]['Info']
            
            for frame_idx, frame_data in case_data.items():
                if 'LandMarks' in frame_data:
                    landmarks_data[str(frame_idx)] = {
                        'landmarks': frame_data['LandMarks'].tolist(),
                        'filename': frame_data['filename'],
                        'frame_number': frame_data['frame_number'],
                        'series_number': frame_data['SeriesNumber']
                    }
        
        metadata = {
            'case_number': self.case_number,
            'series_number': self.series_number,
            'display_range': self.display_range,
            'roi_bounds': self.roi_bounds,
            'first_in_series': self.first_in_series,
            'last_landmark_image': self.last_landmark_image,
            'total_frames_with_landmarks': len(landmarks_data),
            'save_timestamp': str(np.datetime64('now'))
        }
        
        output_data = {
            'metadata': metadata,
            'landmarks': landmarks_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Landmarks saved to: {output_file}")
        print(f"  Total frames with landmarks: {len(landmarks_data)}")
        
        return output_file

    def load_landmarks_from_file(self, input_file: str, all_frames: List[Dict]) -> bool:
        if not os.path.exists(input_file):
            print(f"Error: Landmarks file not found: {input_file}")
            return False
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            metadata = input_data.get('metadata', {})
            landmarks_data = input_data.get('landmarks', {})

            self.case_number = metadata.get('case_number', 1)
            self.series_number = metadata.get('series_number', 1)
            self.display_range = tuple(metadata.get('display_range', (0, 500)))
            self.roi_bounds = tuple(metadata.get('roi_bounds', (0, 0, 0, 0)))
            self.first_in_series = metadata.get('first_in_series', 0)
            self.last_landmark_image = metadata.get('last_landmark_image', 0)

            self.initialize_data_structure(self.case_number, self.series_number)
            
            loaded_count = 0
            for frame_idx_str, frame_data in landmarks_data.items():
                frame_idx = int(frame_idx_str)
                landmarks_array = np.array(frame_data['landmarks'])
                
                frame_info = None
                for frame in all_frames:
                    if frame.get('frame_number', -1) == frame_data.get('frame_number', -1):
                        frame_info = frame
                        break
                
                if frame_info is not None:
                    self.store_landmarks(frame_idx, frame_info, landmarks_array)
                    loaded_count += 1
            
            print(f"✓ Landmarks loaded from: {input_file}")
            print(f"  Loaded {loaded_count} frames with landmarks")
            print(f"  Case: {self.case_number}, Series: {self.series_number}")
            print(f"  Display range: {self.display_range}")
            
            return True
            
        except Exception as e:
            print(f"Error loading landmarks file: {e}")
            return False

    def save_landmarks_to_txt(self, output_file: str = "landmarks_data.txt"):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Landmarks Data Export\n")
            f.write("=" * 50 + "\n\n")
            
            if self.case_number in self.data_storage:
                case_data = self.data_storage[self.case_number]['Info']
                
                for frame_idx, frame_data in sorted(case_data.items()):
                    if 'LandMarks' in frame_data:
                        f.write(f"Frame {frame_idx + 1}:\n")
                        f.write(f"  Filename: {frame_data['filename']}\n")
                        f.write(f"  Frame Number: {frame_data['frame_number']}\n")
                        f.write(f"  Series Number: {frame_data['SeriesNumber']}\n")
                        f.write("  Landmarks:\n")
                        
                        landmarks = frame_data['LandMarks']
                        for i, (x, y) in enumerate(landmarks):
                            f.write(f"    Point {i+1}: ({x:.2f}, {y:.2f})\n")
                        
                        f.write("\n")
            
            total_frames = len([f for f in case_data.values() if 'LandMarks' in f])
            f.write("Summary:\n")
            f.write(f"  Total frames with landmarks: {total_frames}\n")
            f.write(f"  Case number: {self.case_number}\n")
            f.write(f"  Series number: {self.series_number}\n")
            f.write(f"  Export time: {str(np.datetime64('now'))}\n")
        
        print(f"✓ Landmarks saved to TXT: {output_file}")
        return output_file

    def visualize_landmarks_from_file(self, all_frames: List[Dict], input_file: str = None):
        if input_file and os.path.exists(input_file):
            print(f"Loading landmarks from: {input_file}")
            success = self.load_landmarks_from_file(input_file, all_frames)
            if not success:
                return
        
        if not self.data_storage or self.case_number not in self.data_storage:
            print("No landmarks data available. Please load landmarks file first.")
            return
        
        case_data = self.data_storage[self.case_number]['Info']
        frames_with_landmarks = [idx for idx, data in case_data.items() if 'LandMarks' in data]
        
        if not frames_with_landmarks:
            print("No frames with landmarks found.")
            return
        
        print(f"\nEntering edit mode with {len(frames_with_landmarks)} frames with landmarks...")
        print("Navigation:")
        print("  Left/Right arrow: Previous/Next frame")
        print("  Click: Adjust existing landmarks")
        print("  ESC: Exit edit mode")
        
        current_idx = frames_with_landmarks[0]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.canvas.manager.set_window_title('Landmarks Edit Mode - Use arrows to navigate, ESC to exit')
        
        current_landmarks = []
        is_adjusting = False
        selected_point = None
        
        def save_current_landmarks():
            nonlocal current_idx, current_landmarks
            if current_landmarks and len(current_landmarks) == 4:
                self.store_landmarks(current_idx, all_frames[current_idx], np.array(current_landmarks))
                print(f"✓ Saved landmarks for frame {current_idx + 1}")
                return True
            return False
        
        def load_current_landmarks():
            nonlocal current_idx, current_landmarks
            if current_idx in self.data_storage.get(self.case_number, {}).get('Info', {}):
                if 'LandMarks' in self.data_storage[self.case_number]['Info'][current_idx]:
                    existing_landmarks = self.data_storage[self.case_number]['Info'][current_idx]['LandMarks']
                    current_landmarks = existing_landmarks.tolist()
                    return True
            current_landmarks = []
            return False
        
        def update_display(idx, message=None):
            nonlocal current_landmarks
            
            ax.clear()
            frame_info = all_frames[idx]
            image = frame_info['image']
            
            ax.imshow(image, cmap='gray', vmin=self.display_range[0], vmax=self.display_range[1])
            if self.roi_bounds:
                x_min, x_max, y_min, y_max = self.roi_bounds
                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_max, y_min])
            
            total_frames = len(all_frames)
            frame_position = frames_with_landmarks.index(idx) + 1 if idx in frames_with_landmarks else 0
            total_with_landmarks = len(frames_with_landmarks)
            
            title_text = f'Frame {idx+1}/{total_frames} (Landmark {frame_position}/{total_with_landmarks}) - {frame_info["filename"]}'
            
            if message:
                title_text += f'\n{message}'
                
            ax.set_title(title_text)
            
            if current_landmarks:
                colors = ['red', 'green', 'blue', 'yellow']
                for i, (x, y) in enumerate(current_landmarks):
                    ax.plot(x, y, 'o', color=colors[i], markersize=6)
                    ax.text(x + 5, y + 5, str(i+1), color='white', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i], alpha=0.7))
            
            plt.draw()
        
        def on_key(event):
            nonlocal current_idx, is_adjusting, selected_point
            
            if event.key == 'right':
                if current_landmarks and len(current_landmarks) == 4:
                    save_current_landmarks()
                # 改为在所有帧之间导航
                current_idx = min(current_idx + 1, len(all_frames) - 1)
                is_adjusting = False
                selected_point = None
                load_current_landmarks()
                has_lm = "✓ Has landmarks" if current_landmarks else "No landmarks - click to add"
                update_display(current_idx, has_lm)

            elif event.key == 'left':
                if current_landmarks and len(current_landmarks) == 4:
                    save_current_landmarks()
                # 改为在所有帧之间导航
                current_idx = max(current_idx - 1, 0)
                is_adjusting = False
                selected_point = None
                load_current_landmarks()
                has_lm = "✓ Has landmarks" if current_landmarks else "No landmarks - click to add"
                update_display(current_idx, has_lm)    

            elif event.key == 'escape':
                if current_landmarks and len(current_landmarks) == 4:
                    save_current_landmarks()
                plt.close(fig)
                return
        
        def on_click(event):
            nonlocal current_landmarks, is_adjusting, selected_point
            
            if event.inaxes != ax:
                return
            
            if event.button == 1:  
                x, y = event.xdata, event.ydata
                
                min_dist = float('inf')
                clicked_point = None
                
                for i, (lx, ly) in enumerate(current_landmarks):
                    dist = np.sqrt((x - lx)**2 + (y - ly)**2)
                    if dist < 10 and dist < min_dist:  
                        min_dist = dist
                        clicked_point = i
                
                if clicked_point is not None:
                    is_adjusting = True
                    selected_point = clicked_point
                    print(f"Adjusting landmark {selected_point + 1}")
                    return
                
        
        def on_motion(event):
            nonlocal current_landmarks, selected_point
            
            if event.inaxes != ax or selected_point is None:
                return
            
            if selected_point is not None and event.xdata is not None and event.ydata is not None:
                current_landmarks[selected_point] = [event.xdata, event.ydata]
                update_display(current_idx, f"Adjusting landmark {selected_point + 1}")
        
        def on_release(event):
            nonlocal is_adjusting, selected_point
            
            if event.button == 1 and selected_point is not None:
                save_current_landmarks()
                print(f"✓ Adjusted landmark {selected_point + 1} for frame {current_idx + 1}")
                is_adjusting = False
                selected_point = None
                update_display(current_idx, "✓ Landmarks updated")
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)

        load_current_landmarks()
        update_display(current_idx)
        plt.tight_layout()
        plt.show()
        
        response = input("\nSave changes to landmarks file? (y/n): ").strip().lower()
        if response == 'y':
            self.save_landmarks_to_file()
            self.save_landmarks_to_txt()
            print("✓ Landmarks saved to file")
        
        print(f"\nEdit mode completed. Processed {len(frames_with_landmarks)} frames with landmarks.")

    def save_original_without_landmarks(self, all_frames: List[Dict], output_folder: str = "original_without_landmarks"):
        import os
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nSaving original images without landmarks to {output_folder}...")
        
        # 获取第一帧的实际尺寸
        first_frame = all_frames[0]['image']
        original_height, original_width = first_frame.shape
        print(f"Using original image size: {original_width} x {original_height}")
        
        saved_count = 0
        for i in range(len(all_frames)):
            if i in self.data_storage.get(self.case_number, {}).get('Info', {}):
                if 'LandMarks' in self.data_storage[self.case_number]['Info'][i]:
                    frame_info = all_frames[i]
                    image = frame_info['image']
                    
                    current_height, current_width = image.shape
                    if current_height != original_height or current_width != original_width:
                        print(f"Warning: Frame {i+1} size {current_width}x{current_height} differs from first frame {original_width}x{original_height}")
                    
                    dpi = 100
                    fig_width = original_width / dpi
                    fig_height = original_height / dpi
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
                    
                    ax.imshow(image, cmap='gray', 
                            vmin=self.display_range[0], 
                            vmax=self.display_range[1])
                    
                    ax.axis('off')
                    ax.set_position([0, 0, 1, 1]) 
                    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    
                    output_filename = f"original_{i+1:03d}.tif"
                    output_path = os.path.join(output_folder, output_filename)
                    plt.savefig(output_path, dpi=dpi, bbox_inches=None, pad_inches=0)
                    plt.close()
                    
                    saved_count += 1
                    if saved_count % 10 == 0:
                        print(f"  Saved {saved_count} original images...")
        
        print(f"✓ Saved {saved_count} original images without landmarks to '{output_folder}'")
        print(f"  Image size: {original_width} x {original_height}")

    def save_original_with_landmarks(self, all_frames: List[Dict], output_folder: str = "original_with_landmarks"):
        import os
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nSaving original images with landmarks to {output_folder}...")
        
        # 获取第一帧的实际尺寸
        first_frame = all_frames[0]['image']
        original_height, original_width = first_frame.shape
        print(f"Using original image size: {original_width} x {original_height}")
        
        saved_count = 0
        for i in range(len(all_frames)):
            if i in self.data_storage.get(self.case_number, {}).get('Info', {}):
                if 'LandMarks' in self.data_storage[self.case_number]['Info'][i]:
                    frame_info = all_frames[i]
                    image = frame_info['image']
                    landmarks = self.data_storage[self.case_number]['Info'][i]['LandMarks']
                    
                    # 使用实际图像尺寸创建figure
                    dpi = 100
                    fig_width = original_width / dpi
                    fig_height = original_height / dpi
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
                    
                    # 显示图像和标志点 - 移除标题
                    ax.imshow(image, cmap='gray', 
                            vmin=self.display_range[0], 
                            vmax=self.display_range[1])
                    
                    # 绘制标志点
                    colors = ['red', 'green', 'blue', 'yellow']
                    for j, (x, y) in enumerate(landmarks):
                        ax.plot(x, y, 'o', color=colors[j], markersize=6)
                        ax.text(x + 5, y + 5, str(j+1), color='white', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[j], alpha=0.7))
                    
                    # 关键：移除边距、坐标轴和标题
                    ax.axis('off')
                    ax.set_position([0, 0, 1, 1])
                    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    
                    # 保存图片
                    output_filename = f"original_landmarks_{i+1:03d}.tif"
                    output_path = os.path.join(output_folder, output_filename)
                    plt.savefig(output_path, dpi=dpi, bbox_inches=None, pad_inches=0)
                    plt.close()
                    
                    saved_count += 1
                    if saved_count % 10 == 0:
                        print(f"  Saved {saved_count} original images with landmarks...")
        
        print(f"✓ Saved {saved_count} original images with landmarks to '{output_folder}'")
        print(f"  Image size: {original_width} x {original_height}")

    def save_corrected_without_landmarks(self, all_frames: List[Dict], output_folder: str = "corrected_without_landmarks"):
        import os
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nSaving corrected images without landmarks to {output_folder}...")
        
        if self.first_in_series is None:
            print("No reference frame set. Skipping corrected images.")
            return
        
        # 获取第一帧的实际尺寸
        first_frame = all_frames[0]['image']
        original_height, original_width = first_frame.shape
        print(f"Using original image size: {original_width} x {original_height}")
        
        saved_count = 0
        frames_with_landmarks = []
        for i in range(len(all_frames)):
            if i in self.data_storage.get(self.case_number, {}).get('Info', {}):
                if 'LandMarks' in self.data_storage[self.case_number]['Info'][i]:
                    frames_with_landmarks.append(i)
        
        print(f"Found {len(frames_with_landmarks)} frames with landmarks")
        
        for i in frames_with_landmarks:
            # 获取当前帧和参考帧
            current_frame_info = all_frames[i]
            ref_frame_info = all_frames[self.first_in_series]
            
            current_image = current_frame_info['image']
            current_landmarks = self.data_storage[self.case_number]['Info'][i]['LandMarks']
            ref_landmarks = self.data_storage[self.case_number]['Info'][self.first_in_series]['LandMarks']
            
            # 计算变换参数
            params = self.calculate_transformation_matlab_consistent(
                ref_landmarks, current_landmarks, current_image.shape)
            
            # 应用修正
            corrected_image = self.apply_correction_rotation_first(current_image, params)
            
            # 使用实际图像尺寸创建figure
            dpi = 100
            fig_width = original_width / dpi
            fig_height = original_height / dpi
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
            
            ax.imshow(corrected_image, cmap='gray',
                    vmin=self.display_range[0], 
                    vmax=self.display_range[1])
            
            # 关键：移除边距、坐标轴和标题
            ax.axis('off')
            ax.set_position([0, 0, 1, 1])
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            # 保存图片
            output_filename = f"corrected_{i+1:03d}.tif"
            output_path = os.path.join(output_folder, output_filename)
            plt.savefig(output_path, dpi=dpi, bbox_inches=None, pad_inches=0)
            plt.close()
            
            saved_count += 1
            if saved_count % 10 == 0:
                print(f"  Saved {saved_count} corrected images...")
        
        print(f"✓ Saved {saved_count} corrected images without landmarks to '{output_folder}'")
        print(f"  Image size: {original_width} x {original_height}")

    def save_corrected_with_landmarks(self, all_frames: List[Dict], output_folder: str = "corrected_with_landmarks"):
        """保存修正后的图片（带参考标志点）- 使用实际图像尺寸"""
        import os
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nSaving corrected images with reference landmarks to {output_folder}...")
        
        if self.first_in_series is None:
            print("No reference frame set. Skipping corrected images with landmarks.")
            return
        
        # 获取第一帧的实际尺寸
        first_frame = all_frames[0]['image']
        original_height, original_width = first_frame.shape
        print(f"Using original image size: {original_width} x {original_height}")
        
        saved_count = 0
        frames_with_landmarks = []
        for i in range(len(all_frames)):
            if i in self.data_storage.get(self.case_number, {}).get('Info', {}):
                if 'LandMarks' in self.data_storage[self.case_number]['Info'][i]:
                    frames_with_landmarks.append(i)
        
        print(f"Found {len(frames_with_landmarks)} frames with landmarks")

        for i in frames_with_landmarks:
            # 获取当前帧和参考帧
            current_frame_info = all_frames[i]
            ref_frame_info = all_frames[self.first_in_series]
            
            current_image = current_frame_info['image']
            current_landmarks = self.data_storage[self.case_number]['Info'][i]['LandMarks']
            ref_landmarks = self.data_storage[self.case_number]['Info'][self.first_in_series]['LandMarks']
            
            # 计算变换参数
            params = self.calculate_transformation_matlab_consistent(
                ref_landmarks, current_landmarks, current_image.shape)
            
            # 应用修正
            corrected_image = self.apply_correction_rotation_first(current_image, params)
            
            # 使用实际图像尺寸创建figure
            dpi = 100
            fig_width = original_width / dpi
            fig_height = original_height / dpi
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
            
            ax.imshow(corrected_image, cmap='gray',
                    vmin=self.display_range[0], 
                    vmax=self.display_range[1])
            
            # 绘制参考标志点
            colors = ['red', 'green', 'blue', 'yellow']
            for j, (x, y) in enumerate(ref_landmarks):
                ax.plot(x, y, 'o', color=colors[j], markersize=6)
                ax.text(x + 5, y + 5, str(j+1), color='white', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[j], alpha=0.7))
            
            # 关键：移除边距、坐标轴和标题
            ax.axis('off')
            ax.set_position([0, 0, 1, 1])
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            # 保存图片
            output_filename = f"corrected_landmarks_{i+1:03d}.tif"
            output_path = os.path.join(output_folder, output_filename)
            plt.savefig(output_path, dpi=dpi, bbox_inches=None, pad_inches=0)
            plt.close()
            
            saved_count += 1
            if saved_count % 10 == 0:
                print(f"  Saved {saved_count} corrected images with landmarks...")
        
        print(f"✓ Saved {saved_count} corrected images with reference landmarks to '{output_folder}'")
        print(f"  Image size: {original_width} x {original_height}")

    def initialize_data_structure(self, case_number: int, series_number: int):
        self.case_number = case_number
        self.series_number = series_number
        
        if case_number not in self.data_storage:
            self.data_storage[case_number] = {'Info': {}}
    
    def store_landmarks(self, image_idx: int, image_info: Dict, landmarks: np.ndarray):
        self.data_storage[self.case_number]['Info'][image_idx] = {
            'LandMarks': landmarks,
            'SeriesNumber': self.series_number,
            'image_data': image_info['image'],
            'filename': image_info['filename'],
            'frame_number': image_info.get('frame_number', image_idx + 1)
        }
    
    def copy_landmarks_from_previous_frame(self, current_idx: int, all_frames: List[Dict]) -> bool:
        if current_idx == 0:
            return False  # 第一帧没有上一帧
        
        prev_idx = current_idx - 1
        if prev_idx not in self.data_storage.get(self.case_number, {}).get('Info', {}):
            return False
        
        # 直接复制上一帧的landmarks位置
        prev_landmarks = self.data_storage[self.case_number]['Info'][prev_idx]['LandMarks']
        self.store_landmarks(current_idx, all_frames[current_idx], prev_landmarks.copy())
        
        print(f"✓ Copied landmarks from frame {prev_idx + 1} to frame {current_idx + 1}")
        return True

    def interactive_navigation_collect_landmarks(self, all_frames: List[Dict]):
        print("\n=== Interactive Landmark Collection ===")
        print("Navigation:")
        print("  Left/Right arrow: Previous/Next frame")
        print("  c: Copy landmarks from previous frame")
        print("  Click: Collect landmarks (4 points) or adjust existing points")
        print("  d: Delete current frame landmarks")
        print("  s: Save landmarks to file")
        print("  l: Load landmarks from file")
        print("  ESC: Exit landmark collection")
        
        self.first_in_series = 0
        current_idx = 0
        
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.canvas.manager.set_window_title('Landmark Collection - Use arrows to navigate, ESC to exit')
        
        landmarks_collected = 0
        current_landmarks = []
        is_adjusting = False
        selected_point = None
        
        def save_current_landmarks():
            nonlocal current_idx, current_landmarks
            if current_landmarks and len(current_landmarks) == 4:
                self.store_landmarks(current_idx, all_frames[current_idx], np.array(current_landmarks))
                print(f"✓ Saved landmarks for frame {current_idx + 1}")
                return True
            return False
        
        def load_current_landmarks():
            nonlocal current_idx, current_landmarks, landmarks_collected
            print(f"Loading frame {current_idx + 1}, keys in storage: {sorted(self.data_storage.get(self.case_number, {}).get('Info', {}).keys())[-5:]}")
            if current_idx in self.data_storage.get(self.case_number, {}).get('Info', {}):
                if 'LandMarks' in self.data_storage[self.case_number]['Info'][current_idx]:
                    existing_landmarks = self.data_storage[self.case_number]['Info'][current_idx]['LandMarks']
                    current_landmarks = existing_landmarks.tolist()
                    landmarks_collected = len(current_landmarks)
                    print(f"  → Loaded {landmarks_collected} landmarks")
                    return True
            current_landmarks = []
            landmarks_collected = 0
            print(f"  → No landmarks found for frame {current_idx + 1}")
            return False
        
        def update_display(idx, message=None):
            nonlocal landmarks_collected, current_landmarks
            
            ax.clear()
            frame_info = all_frames[idx]
            image = frame_info['image']
            
            ax.imshow(image, cmap='gray', vmin=self.display_range[0], vmax=self.display_range[1])
            if self.roi_bounds:
                x_min, x_max, y_min, y_max = self.roi_bounds
                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_max, y_min])
            
            # 显示当前帧信息
            total_frames = len(all_frames)
            title_text = f'Frame {idx+1}/{total_frames} - {frame_info["filename"]}'
            
            if message:
                title_text += f'\n{message}'
                
            ax.set_title(title_text)
            
            # 显示已存在的标志点（从数据存储中读取）
            if idx in self.data_storage.get(self.case_number, {}).get('Info', {}):
                if 'LandMarks' in self.data_storage[self.case_number]['Info'][idx]:
                    existing_landmarks = self.data_storage[self.case_number]['Info'][idx]['LandMarks']
                    colors = ['red', 'green', 'blue', 'yellow']
                    for i, (x, y) in enumerate(existing_landmarks):
                        ax.plot(x, y, 'o', color=colors[i], markersize=6)
                        ax.text(x + 5, y + 5, str(i+1), color='white', fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i], alpha=0.7))
            
            # 显示当前正在收集的标志点
            if current_landmarks:
                landmarks_array = np.array(current_landmarks)
                ax.plot(landmarks_array[:, 0], landmarks_array[:, 1], 'og', markersize=6, alpha=0.7)
                colors = ['red', 'green', 'blue', 'yellow']
                for i, (x, y) in enumerate(current_landmarks):
                    ax.text(x + 5, y + 5, str(i+1), color='white', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i], alpha=0.7))
            
            plt.draw()
        
        def on_key(event):
            print(f"Key pressed: {event.key}")  # ← 加这一行在最开头
            nonlocal current_idx, landmarks_collected, current_landmarks, is_adjusting, selected_point
            
            if event.key == 'right':
                print(f"Right pressed, current: {current_idx}, max: {len(all_frames)-1}")
                # 保存当前帧的修改
                if current_landmarks and len(current_landmarks) == 4:
                    save_current_landmarks()
                
                # 移动到下一帧
                current_idx = min(current_idx + 1, len(all_frames) - 1)
                landmarks_collected = 0
                is_adjusting = False
                selected_point = None
                
                # 加载新帧的landmarks
                load_current_landmarks()
                update_display(current_idx, f"Loaded frame {current_idx + 1}")
                
            elif event.key == 'left':
                # 保存当前帧的修改
                if current_landmarks and len(current_landmarks) == 4:
                    save_current_landmarks()
                
                # 移动到上一帧
                current_idx = max(current_idx - 1, 0)
                landmarks_collected = 0
                is_adjusting = False
                selected_point = None
                
                # 加载新帧的landmarks
                load_current_landmarks()
                update_display(current_idx, f"Loaded frame {current_idx + 1}")
                
            elif event.key == 'escape':
                # 退出前保存当前帧
                if current_landmarks and len(current_landmarks) == 4:
                    save_current_landmarks()
                plt.close(fig)
                return
                
            elif event.key == 'c':
                print(f"Current idx: {current_idx}")
                print(f"Previous idx: {current_idx - 1}")
                print(f"Keys in storage: {sorted(self.data_storage.get(self.case_number, {}).get('Info', {}).keys())}")
                if self.copy_landmarks_from_previous_frame(current_idx, all_frames):
                    load_current_landmarks()
                    update_display(current_idx, "✓ Copied landmarks from previous frame")
                else:
                    update_display(current_idx, "No previous landmarks to copy")
                    
            elif event.key == 'd':  # 删除当前帧的landmarks
                if current_idx in self.data_storage.get(self.case_number, {}).get('Info', {}):
                    del self.data_storage[self.case_number]['Info'][current_idx]
                    current_landmarks = []
                    landmarks_collected = 0
                    update_display(current_idx, "✓ Landmarks deleted")
                    print(f"✓ Deleted landmarks for frame {current_idx + 1}")
            
            elif event.key == 's':  # 保存到文件
                if current_landmarks and len(current_landmarks) == 4:
                    save_current_landmarks()
                self.save_landmarks_to_file()
                self.save_landmarks_to_txt()
                update_display(current_idx, "✓ Landmarks saved to file")
            
            elif event.key == 'l':  # 从文件加载
                filename = input("Enter landmarks filename (or press Enter for landmarks_data.json): ").strip()
                if not filename:
                    filename = "landmarks_data.json"
                if self.load_landmarks_from_file(filename, all_frames):
                    load_current_landmarks()
                    update_display(current_idx, f"✓ Landmarks loaded from {filename}")
                else:
                    update_display(current_idx, f"Failed to load from {filename}")
        
        def on_click(event):
            nonlocal landmarks_collected, current_landmarks, is_adjusting, selected_point
            
            if event.inaxes != ax:
                return
            
            if event.button == 1:  # 左键点击
                x, y = event.xdata, event.ydata
                
                # 检查是否点击了现有的landmark（从当前显示的landmarks中检查）
                min_dist = float('inf')
                clicked_point = None
                
                for i, (lx, ly) in enumerate(current_landmarks):
                    dist = np.sqrt((x - lx)**2 + (y - ly)**2)
                    if dist < 10 and dist < min_dist:  # 10像素内的点击阈值
                        min_dist = dist
                        clicked_point = i
                
                if clicked_point is not None:
                    # 进入调整模式
                    is_adjusting = True
                    selected_point = clicked_point
                    print(f"Adjusting landmark {selected_point + 1}")
                    return
                
                # 如果不是调整模式，正常收集landmarks
                if not is_adjusting:
                    if landmarks_collected < 4:
                        if landmarks_collected < len(current_landmarks):
                            # 替换现有的点
                            current_landmarks[landmarks_collected] = [x, y]
                        else:
                            # 添加新点
                            current_landmarks.append([x, y])
                        
                        landmarks_collected += 1
                        
                        print(f"Landmark {landmarks_collected}: ({x:.1f}, {y:.1f})")
                        
                        # 如果收集了4个点，自动保存
                        if landmarks_collected == 4:
                            save_current_landmarks()
                            update_display(current_idx, "✓ Landmarks saved!")
                        else:
                            update_display(current_idx)
        
        def on_motion(event):
            nonlocal current_landmarks, selected_point
            
            if event.inaxes != ax or selected_point is None:
                return
            
            # 在调整模式下移动选中的点
            if selected_point is not None and event.xdata is not None and event.ydata is not None:
                current_landmarks[selected_point] = [event.xdata, event.ydata]
                update_display(current_idx, f"Adjusting landmark {selected_point + 1}")
        
        def on_release(event):
            nonlocal is_adjusting, selected_point
            
            if event.button == 1 and selected_point is not None:
                # 调整完成，保存修改
                save_current_landmarks()
                print(f"✓ Adjusted landmark {selected_point + 1} for frame {current_idx + 1}")
                is_adjusting = False
                selected_point = None
                update_display(current_idx, "✓ Landmarks updated")
        
        # 连接事件
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        
        # 初始显示：加载第一帧的landmarks
        load_current_landmarks()
        update_display(current_idx)
        plt.tight_layout()
        plt.show()
        
        self.last_landmark_image = current_idx
        self.first_in_series = 0  # 假设第一帧是参考帧
        
        print(f"\nLandmark collection completed. Processed {current_idx + 1} frames.")
        
        # 询问是否保存landmarks到文件
        response = input("\nSave landmarks to file? (y/n): ").strip().lower()
        if response == 'y':
            self.save_landmarks_to_file()
            self.save_landmarks_to_txt()
    
    def calculate_transformation_matlab_consistent(self, ref_landmarks: np.ndarray, 
                                                 tar_landmarks: np.ndarray, 
                                                 image_shape: Tuple[int, int]) -> np.ndarray:
        center = np.array([image_shape[1]/2, image_shape[0]/2])
        
        ref_pts_centered = ref_landmarks - center
        tar_pts_centered = tar_landmarks - center
        
        initial_guess = [0, 0, 0]
        result = minimize(self.sum_sqrd_error, initial_guess,
                         args=(ref_pts_centered, tar_pts_centered),
                         method='Nelder-Mead',
                         options={'maxiter': 1000})
        
        if result.success:
            return result.x
        else:
            return np.array([0, 0, 0])  # 返回零变换而不是抛出异常

    def apply_correction_rotation_first(self, image: np.ndarray, params: np.ndarray) -> np.ndarray:
        dx, dy, theta = params
        height, width = image.shape
        
        # 先旋转
        M_rotate = cv2.getRotationMatrix2D((width/2, height/2), theta, 1.0)
        rotated = cv2.warpAffine(image, M_rotate, (width, height), 
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # 后平移
        M_translate = np.float32([[1, 0, dx], [0, 1, dy]])
        corrected = cv2.warpAffine(rotated, M_translate, (width, height),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return corrected

    def create_correction_video(self, all_frames: List[Dict], output_path: str = "correction_video.mp4"):
        if self.first_in_series is None or self.last_landmark_image is None:
            print("No landmarks collected yet. Please run landmark collection first.")
            return
        
        print("\nCreating correction video...")
        
        # 获取实际图像尺寸
        sample_image = all_frames[0]['image']
        height, width = sample_image.shape
        print(f"Video size: {width} x {height}")
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 5
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            for i in range(self.first_in_series, min(self.last_landmark_image + 1, len(all_frames))):
                if i in self.data_storage.get(self.case_number, {}).get('Info', {}):
                    if 'LandMarks' in self.data_storage[self.case_number]['Info'][i]:
                        # 获取当前帧和参考帧
                        current_frame_info = all_frames[i]
                        ref_frame_info = all_frames[self.first_in_series]
                        
                        current_image = current_frame_info['image']
                        ref_image = ref_frame_info['image']
                        
                        # 获取标志点
                        ref_landmarks = self.data_storage[self.case_number]['Info'][self.first_in_series]['LandMarks']
                        current_landmarks = self.data_storage[self.case_number]['Info'][i]['LandMarks']
                        
                        # 计算变换参数
                        params = self.calculate_transformation_matlab_consistent(
                            ref_landmarks, current_landmarks, current_image.shape)
                        
                        # 应用修正
                        corrected_image = self.apply_correction_rotation_first(current_image, params)
                        
                        # 准备显示图像
                        display_img = cv2.normalize(corrected_image, None, 0, 255, cv2.NORM_MINMAX)
                        display_img = display_img.astype(np.uint8)
                        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                        
                        # 添加参考标志点
                        for point in ref_landmarks:
                            x, y = int(point[0]), int(point[1])
                            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)  # 红色点
                        
                        # 添加帧信息文本
                        cv2.putText(display_img, f'Frame {i+1}', (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(display_img, f'dx: {params[0]:.1f} dy: {params[1]:.1f} θ: {params[2]:.1f}°', 
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        video_writer.write(display_img)
                        
                        if (i - self.first_in_series) % 10 == 0:
                            print(f"Processed frame {i+1} for video")
            
            print(f"✓ Video saved to: {output_path}")
            print(f"  Video size: {width} x {height}")
            
        finally:
            video_writer.release()

    def process_2d_sequence(self, dicom_files: List[Dict]) -> List[Dict]:
        """专门处理2D DICOM序列"""
        all_frames = []
        
        print(f"Processing 2D sequence with {len(dicom_files)} images...")
        
        # 按实例号排序
        sorted_files = sorted(dicom_files, key=lambda x: x['instance_number'])
        
        for i, file_info in enumerate(sorted_files):
            frame_info = file_info.copy()
            frame_info['frame_number'] = i + 1
            frame_info['total_frames'] = len(sorted_files)
            frame_info['is_2d_sequence'] = True
            all_frames.append(frame_info)
        
        print(f"✓ Prepared {len(all_frames)} frames for processing")
        return all_frames

    def enhanced_process_3d_series(self, folder_path: str, case_number: int = 1, series_number: int = 1):
        """增强的处理流程，支持2D序列和3D数据"""
        print("=" * 60)
        print("Enhanced DICOM Processing (2D Sequence & 3D Series)")
        print("=" * 60)
        
        # 初始化数据结构
        self.initialize_data_structure(case_number, series_number)
        
        # 加载DICOM文件
        all_dicom_files = self.load_dicom_series(folder_path)
        series_dict = self.analyze_series(all_dicom_files)
        selected_files = self.select_series_interactive(series_dict)

        # 检查是否是2D序列
        selected_series_key = list(series_dict.keys())[0]  # 假设选择了第一个系列
        is_2d_sequence = series_dict[selected_series_key].get('is_2d_sequence', False)
        
        if is_2d_sequence:
            print("✓ Processing as 2D DICOM sequence")
            # 对于2D序列，直接使用文件列表
            all_frames = self.extract_frames_from_3d(selected_files)
        else:
            print("✓ Processing as 3D DICOM series") 
            all_frames = self.extract_frames_from_3d(selected_files)
        
        # 显示详细的图像尺寸信息
        print("\n=== DICOM Image Size Analysis ===")
        unique_sizes = {}
        for i, frame_info in enumerate(all_frames):
            image = frame_info['image']
            size_key = image.shape
            if size_key not in unique_sizes:
                unique_sizes[size_key] = []
            unique_sizes[size_key].append(i)
        
        print(f"Found {len(unique_sizes)} unique image size(s):")
        for size, frames in unique_sizes.items():
            print(f"  Size {size[1]}x{size[0]}: {len(frames)} frames (first frame: {frames[0]+1})")
        
        # 检查所有帧尺寸是否一致
        if len(unique_sizes) > 1:
            print("Warning: Multiple image sizes detected in the series!")
            # 使用第一帧的尺寸作为基准
            first_size = all_frames[0]['image'].shape
            print(f"Using first frame size {first_size[1]}x{first_size[0]} as reference")
        else:
            first_size = list(unique_sizes.keys())[0]
            print(f"All frames have consistent size: {first_size[1]}x{first_size[0]}")

        # 让用户选择处理哪些帧
        print(f"\nTotal frames available: {len(all_frames)}")
        print("Select frames to process:")
        print("  Enter range like '1-50' for frames 1 to 50")
        print("  Press Enter to process all frames")

        frame_input = input("Frame selection: ").strip()

        if frame_input == '':
            # 处理所有帧
            selected_indices = list(range(len(all_frames)))
            print(f"Processing all {len(all_frames)} frames")

        elif '-' in frame_input and ',' not in frame_input:
            # 范围选择，如 '1-50'
            try:
                start, end = frame_input.split('-')
                start = max(0, int(start) - 1)        # 转为0-index
                end = min(len(all_frames), int(end))   # 包含end帧
                selected_indices = list(range(start, end))
                print(f"Processing frames {start+1} to {end} ({len(selected_indices)} frames)")
            except ValueError:
                print("Invalid range, processing all frames")
                selected_indices = list(range(len(all_frames)))

        # 按选择过滤帧
        all_frames = [all_frames[i] for i in selected_indices]
        print(f"✓ {len(all_frames)} frames selected for processing")

        # 设置显示范围
        first_image = all_frames[0]['image']
        self.adjust_display_range(first_image)
        
        # 选择ROI
        self.roi_bounds = self.select_roi(first_image)
        
        print("\n=== Processing Steps ===")
        print("1. Interactive landmark collection (copy from previous frame)")
        print("2. Motion correction and video generation") 
        print("3. Save original images without landmarks")
        print("4. Save original images with landmarks") 
        print("5. Save corrected images without landmarks")
        print("6. Save corrected images with reference landmarks")
        print("7. Save landmarks to file for future use")
        
        # 询问是否从文件加载已有的landmarks
        response = input("\nLoad existing landmarks from file? (y/n): ").strip().lower()
        if response == 'y':
            filename = input("Enter landmarks filename (or press Enter for landmarks_data.json): ").strip()
            if not filename:
                filename = "landmarks_data.json"
            if self.load_landmarks_from_file(filename, all_frames):
                print("✓ Landmarks loaded successfully!")
                print(f"  Loaded landmarks for frames 1-100, total frames available: {len(all_frames)}")
                mode = input("Choose mode:\n  1: Edit existing landmarks only (frames 1-100)\n  2: Continue adding new landmarks (frames 101+)\n  Enter 1 or 2: ").strip()
                if mode == '1':
                    self.visualize_landmarks_from_file(all_frames)
                else:
                    # 进入收集模式，可以访问所有帧
                    print("Entering collection mode - you can navigate all frames and add new landmarks...")
                    self.interactive_navigation_collect_landmarks(all_frames)
            else:
                print("Starting new landmark collection...")
                # 步骤1: 交互式标志点收集（支持从上一帧复制）
                self.interactive_navigation_collect_landmarks(all_frames)
        else:
            # 步骤1: 交互式标志点收集（支持从上一帧复制）
            self.interactive_navigation_collect_landmarks(all_frames)
        
        # 步骤2: 运动修正和视频生成
        self.create_correction_video(all_frames, "motion_correction_video.mp4")
        
        # 步骤3-6: 分别保存四种类型的图片
        self.save_original_without_landmarks(all_frames, "01_original_without_landmarks")
        self.save_original_with_landmarks(all_frames, "02_original_with_landmarks")
        self.save_corrected_without_landmarks(all_frames, "03_corrected_without_landmarks")
        self.save_corrected_with_landmarks(all_frames, "04_corrected_with_landmarks")
        
        # 步骤7: 保存landmarks到文件
        self.save_landmarks_to_file()
        self.save_landmarks_to_txt()
        
        print("\n" + "=" * 60)
        print("Enhanced Processing Completed!")
        print("=" * 60)
        
        # 显示统计信息
        landmark_frames = len(self.data_storage[case_number]['Info'])
        print(f"\nProcessing Statistics:")
        print(f"Total frames processed: {len(all_frames)}")
        print(f"Frames with landmarks: {landmark_frames}")
        print(f"\nOutput files created:")
        print(f"  01_original_without_landmarks/ - Original images without landmarks")
        print(f"  02_original_with_landmarks/    - Original images with current landmarks") 
        print(f"  03_corrected_without_landmarks/ - Corrected images without landmarks")
        print(f"  04_corrected_with_landmarks/    - Corrected images with reference landmarks")
        print(f"  motion_correction_video.mp4     - Correction video")
        print(f"  landmarks_data.json            - Landmarks data (JSON)")
        print(f"  landmarks_data.txt             - Landmarks data (human-readable)")
        
        return all_frames

def main():
    corrector = Weasis3DMotionCorrection()
    
    #folder_path = "/hpc/lsu876/MRI_seg/results_motionCorrection_and_contraction map/motion correction/MRB015/273f3eae/5a30e1e4/5a30e63a"  
    folder_path = "./273f3eae/5a30e1e4/5a30e63a" 
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return
    
    try:  
        # 使用增强的处理流程
        all_frames = corrector.enhanced_process_3d_series(folder_path, case_number=1, series_number=1)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()