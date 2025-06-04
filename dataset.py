import os
import cv2
import numpy as np
import shutil
from pathlib import Path

def xdog(img_gray, sigma=0.8, k=3, gamma=1, epsilon=0.001, phi=10.0):
    img = img_gray.astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(img, (0, 0), sigma)
    g2 = cv2.GaussianBlur(img, (0, 0), sigma * k)
    D = g1 - gamma * g2

    xdog = 1.0 + np.tanh(phi * (D - epsilon))
    xdog = (xdog * 255).clip(0, 255).astype(np.uint8)

    return xdog

def process_and_save(img_path, save_dir):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError("图像无法读取")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        xdog_img = xdog(gray)

        # 将单通道的 xdog 图转换为 BGR，以便与原图拼接
        xdog_bgr = cv2.cvtColor(xdog_img, cv2.COLOR_GRAY2BGR)

        # 拼接原图 + xdog图
        combined = np.concatenate((img, xdog_bgr), axis=1)

        # 确保输出文件夹存在
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / img_path.name
        cv2.imwrite(str(save_path), combined)

        # 删除原图
        img_path.unlink()
    except Exception as e:
        print(f"跳过损坏或错误图像: {img_path.name}, 错误: {e}")
        try:
            img_path.unlink()
        except:
            pass

def process_all():
    src_dirs = ['train', 'val']
    target_root = Path('dataset')

    for src in src_dirs:
        src_root = Path(src)
        save_root = target_root / src
        for img_path in src_root.rglob('*.*'):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            process_and_save(img_path, save_root)

        if src_root.exists():
            shutil.rmtree(src_root, ignore_errors=True)

if __name__ == "__main__":
    process_all()
    print("全部处理完成。")
