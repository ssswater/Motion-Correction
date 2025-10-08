# Gastric MRI Motion Correction and Evaluation Toolkit

This repository provides a collection of Python scripts for **motion correction**, **quantitative evaluation**, and **visualization** of 2D gastric MRI data.  
It supports multiple motion correction methods and provides tools to assess the correction quality using DVARS metrics and visualization outputs.

---

## Overview

| Script                                    | Function                                                     |
| ----------------------------------------- | ------------------------------------------------------------ |
| **motionCorrection.py**                   | Performs motion correction on dynamic gastric MRI images using three methods: <br>1️⃣ ECC only (intensity-based registration) <br>2️⃣ B-spline only (deformation-based registration) <br>3️⃣ ECC followed by B-spline (combined approach, recommended) |
| **extractVerticleLine_calculateDVARS.py** | Extracts intensity profiles along a vertical line through the stomach and computes **DVARS** (temporal signal variance) to evaluate correction performance. |
| **analysisDVARS.py**                      | Performs statistical comparison of DVARS metrics across multiple cases, before and after motion correction. |
| **makeVideo.py**                          | Converts sequential image frames (e.g., from MRI time series or correction outputs) into a video file for visualization. |

---

## Motion Correction Methods

The `motionCorrection.py` script provides three selectable correction strategies:

1. **ECC only**  
   - Uses Enhanced Correlation Coefficient–based registration (OpenCV implementation).  
   - Suitable for small, global motions.

2. **B-spline only**  
   - Uses deformable registration based on B-spline transformation.  
   - Suitable for local deformations and tissue movement.

3. **ECC + B-spline (recommended)**  
   - Applies global ECC alignment first, followed by local B-spline refinement.  
   - Produces the best results for gastric MRI motion correction.

