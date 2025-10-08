import cv2
import os
import glob

def tif_to_video(input_folder, output_video_path, fps=30):
    # 获取所有.tif文件并按文件名排序
    tif_files = sorted(glob.glob(os.path.join(input_folder, "*.tif")))
    
    if not tif_files:
        raise ValueError("未找到TIFF文件！")

    # 读取第一张图片获取尺寸
    first_frame = cv2.imread(tif_files[0], cv2.IMREAD_UNCHANGED)
    height, width = first_frame.shape[:2]

    # 初始化视频写入器（MP4格式）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'avc1'
    video_writer = cv2.VideoWriter(
        output_video_path, 
        fourcc, 
        fps, 
        (width, height)
    )

    # 逐帧写入视频
    for tif_file in tif_files:
        frame = cv2.imread(tif_file)
        if frame is None:
            print(f"警告：无法读取 {tif_file}，跳过")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"视频已保存至 {output_video_path}")

tif_to_video(
    input_folder="./input_registered_images",
    output_video_path="./output_video.mp4",
    fps=24  # 根据需求调整帧率
)