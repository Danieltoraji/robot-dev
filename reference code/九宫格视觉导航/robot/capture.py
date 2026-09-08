import subprocess
import time
import os
import cv2   # 别忘了导入 opencv

def capture_image():
    """
    调用 fswebcam 拍照，保存到 ~/Pictures，并返回图像数组 (BGR格式)
    返回: numpy.ndarray 或 None（失败时）
    """
    # 1. 定位到 template 文件夹，不存在则自动创建
    pictures_dir = os.path.expanduser("~/Pictures")
    os.makedirs(pictures_dir, exist_ok=True)
    
    # 2. 生成唯一文件名（时间戳）
    timestamp = int(time.time())
    filename = os.path.join(pictures_dir, f"photo_{timestamp}.jpg")
    
    # 3. 执行 fswebcam 拍照命令（沿用您原来的参数）
    cmd = f"fswebcam -r 2592x1944 --no-banner -S 3 {filename}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # 4. 检查拍照是否成功
    if result.returncode != 0:
        print(f"❌ 拍照失败: {result.stderr}")
        return None
    
    print(f"✅ 照片已保存: {filename}")
    
    # ===== 核心改动：把保存的文件读进内存 =====
    img = cv2.imread(filename)
    if img is None:
        print("⚠️ 文件已保存，但 OpenCV 无法读取，可能文件损坏")
        return None
    
    # 保持较高分辨率，确保地面数字有足够像素供 OCR 识别
    # 若机器人算力不足，可调低，但建议不要低于 1280x960
    img = cv2.resize(img, (1280,980), interpolation=cv2.INTER_AREA)

    # 5. 返回图像数组（BGR 格式，和之前完全一致）
    return img


