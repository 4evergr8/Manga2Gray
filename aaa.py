import cv2
import numpy as np

def xdog(img_gray, sigma=0.8, k=3.0, gamma=0.95, epsilon=0.001, phi=10.0):
    img = img_gray.astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(img, (0, 0), sigma)
    g2 = cv2.GaussianBlur(img, (0, 0), sigma * k)
    D = g1 - gamma * g2
    xdog = 1.0 + np.tanh(phi * (D - epsilon))
    xdog = (xdog * 255).clip(0, 255).astype(np.uint8)
    return xdog

def test_gamma(image_path, gamma_values, output_path='xdog_gamma_test.png'):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像。")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result_imgs = []

    for gamma in gamma_values:
        xdog_img = xdog(gray, gamma=gamma, sigma=0.8, k=3.0, epsilon=0.001, phi=10.0)
        xdog_img_colored = cv2.putText(
            cv2.cvtColor(xdog_img, cv2.COLOR_GRAY2BGR),
            f'gamma={gamma:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2, cv2.LINE_AA
        )
        result_imgs.append(xdog_img_colored)

    combined = np.concatenate(result_imgs, axis=1)
    cv2.imwrite(output_path, combined)
    print(f"结果已保存至：{output_path}")

if __name__ == '__main__':
    gamma_list = [0.80, 0.90, 0.95, 0.98, 1.00]
    image_file = 'aaa.jpg'  # 替换为测试图路径
    test_gamma(image_file, gamma_list)
