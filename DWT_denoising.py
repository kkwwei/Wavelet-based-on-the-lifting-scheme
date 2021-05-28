import numpy as np
import cv2
import math
def waveleteTransform(img):
    # 将图像像素类型转换成浮点型
    image = img.astype(float)
    height, width = image.shape[:2]
    result = np.zeros((height, width, 3), float)

    # 水平方向第一次处理
    width2 = int(width / 2)
    for i in range(height):

        for j in range(0, width - 1, 2):
            # 分为奇序列和偶序列进行处理
            j1 = (int)(j + 1)
            j2 = (int)(j / 2)
            # 向下取整
            width3 = width2 + j2
            # 采用提升方案
            # xc=(xo + xe)/2
            # xd=(x0 - xd)/2
            # [xc xd]
            result[i, j2] = ((image[i, j] + image[i, j1]) / 2)
            result[i, width3] = ((image[i, j] - image[i, j1]) / 2)


    # copy array
    image = np.copy(result)
    result=np.zeros((height, width, 3), float)
    # 垂直方向第一次处理
    height2 = int(height / 2)
    for i in range(0, height - 1, 2):
        for j in range(0, width):
            i1 = (int)(i + 1)
            i2 = (int)(i / 2)
            height3 = height2 + i2

            result[i2, j] = (image[i, j] + image[i1, j]) / 2
            result[height3, j] = (image[i, j] - image[i1, j]) / 2

    image = np.copy(result).astype(np.uint8)
    HH = np.copy(image[height2+1:,width2+1:])
    LH = np.copy(image[height2+1:,:width2+1])
    HL = np.copy(image[:height2+1,width2+1:])
    LL = np.copy(image[:height2+1,:width2+1])
    return [image,HH,LH,HL,LL]


def denoise(img):
    #采用软阈值法进行去噪
    image = img.astype(float)
    #sigma = abs(np.median(image))/0.6745
    #threshold = math.sqrt(sigma*(2*math.log(len(image))))
    #image[(abs(image)<threshold)]= 0.0
    #image[image > threshold] -= threshold
    #image[image < (-threshold)] += threshold
    image[(abs(image) < 256)] = 0.0
    image = image.astype(np.uint8)
    return image

def inverseWaveleteTransform(img):
    image = img.astype(float)
    nr, nc = image.shape[:2]
    result = np.zeros((nr, nc, 3), float)
    nr2 = nr / 2

    for i in range(0, nr - 1, 2):
        for j in range(0, nc):

            i1 = (int) (i + 1)
            i2 = (int) (i / 2)
            nr3 = (int) (nr2 + i2)

            result[i, j] = ((image[i2, j] / 2) + (image[nr3, j] / 2)) * 2
            result[i1, j] = ((image[i2, j] / 2) - (image[nr3, j] / 2)) * 2

    # //copy array
    image = np.copy(result)

    # // Horizontal processing:
    nc2 = nc / 2
    for i in range(0, nr):
        for j in range(0, nc - 1, 2):

            j1 = (int) (j + 1)
            j2 = (int) (j / 2)
            nc3 = (int) (j2 + nc2)
            result[i, j] = ((image[i, j2] / 2) + (image[i, nc3] / 2)) * 2
            result[i, j1] = ((image[i, j2] / 2) - (image[i, nc3] / 2)) * 2

    resultimg = result.astype(np.uint8)
    return resultimg


if __name__ == '__main__':
    # loadImage & copy image
    image = cv2.imread("./image/image_noise.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./image/gray_image_noise.jpg', image)
    height, width = image.shape[:2]
    image2, HH, LH, HL, LL = waveleteTransform(image)
    cv2.imwrite('./image/DWT.jpg', image2)
    HH_d = denoise(HH)
    HL_d = denoise(HL)
    LH_d = denoise(LH)
    image3 = np.copy(image2)
    #图3是去噪后的分解图

    image3[int(height/2)+1:,int(width/2)+1:] = HH_d;
    image3[int(height/2)+1:,:int(width/2)+1] = LH_d;
    image3[:int(height/2)+1,int(width/2)+1:] = HL_d;
    cv2.imwrite('./image/DWT_denoise.jpg', image3)
    image4 = inverseWaveleteTransform(image2)
    #cv2.imwrite('./image/IDWT.jpg', image4)
    #图像4是重构后的原图
    image5 = inverseWaveleteTransform(image3)
    cv2.imwrite('./image/IDWT_denoise.jpg', image5)
    #图像5是重构后的去噪图像
