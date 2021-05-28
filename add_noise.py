import numpy as np
import cv2
def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        image:原始图像
        mean : 均值
        var : 方差,越大，噪声越大
    '''
    image = image.astype(float)
    image = image/255.0
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    img = image + noise
    img = np.clip(img, 0.0, 1.0)
    img = (img*255).astype(np.uint8)
    return img

if __name__ == '__main__':
    image = cv2.imread("./image/lena.jpg")
    image_noise = gasuss_noise(image, mean=0, var=0.01)
    cv2.imwrite('./image/image_noise.jpg', image_noise)
    cv2.imshow('./image/image_noise', image_noise)
    cv2.waitKey(0)