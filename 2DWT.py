import numpy as np
import cv2


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

    image = np.copy(result)
    result=np.zeros((height2, width2, 3), float)
    # 对LL部分进行第二次DWT变换
    width4 = int(width2 / 2)
    for i in range(height2):

        for j in range(0, width2 - 1, 2):
            # 分为奇序列和偶序列进行处理
            j1 = (int)(j + 1)
            j2 = (int)(j / 2)
            # 向下取整
            width5 = width4 + j2
            # 采用提升方案
            # xc=(xo + xe)/2
            # xd=(x0 - xd)/2
            # [xc xd]
            result[i, j2] = ((image[i, j] + image[i, j1]) / 2)
            result[i, width5] = ((image[i, j] - image[i, j1]) / 2)
    x, y = result.shape[:2]
    image[:x, :y] = result[:, :]
    result=np.zeros((height2, width2, 3), float)
    height4 = int(height2 / 2)
    for i in range(0, height2 - 1, 2):
        for j in range(0, width2):
            i1 = (int)(i + 1)
            i2 = (int)(i / 2)
            height5 = height4 + i2

            result[i2, j] = (image[i, j] + image[i1, j]) / 2
            result[height5, j] = (image[i, j] - image[i1, j]) / 2
    x, y = result.shape[:2]
    image[:x, :y] = result[:, :]

    resultimg = image.astype(np.uint8)
    return resultimg


def inverseWaveleteTransform(img):
    image = img.astype(float)
    nr, nc = image.shape[:2]
    nr2 = int(nr / 2)
    nr3 = int(nr2 / 2)
    nc2 = int(nc / 2)
    nc3 = int(nc2 / 2)
    #第一次
    result = np.zeros((nr2, nc2, 3), float)
    for i in range(0, nr2 - 1, 2):
        for j in range(0, nc2):
            i1 = (int)(i + 1)
            i2 = (int)(i / 2)
            nr4 = (int)(nr3 + i2)

            result[i, j] = ((image[i2, j] / 2) + (image[nr4, j] / 2)) * 2
            result[i1, j] = ((image[i2, j] / 2) - (image[nr4, j] / 2)) * 2
    image[:nr2,:nc2]=result[:,:]
    for i in range(0, nr2):
        for j in range(0, nc2 - 1, 2):
            j1 = (int)(j + 1)
            j2 = (int)(j / 2)
            nc4 = (int)(j2 + nc3)
            result[i, j] = ((image[i, j2] / 2) + (image[i, nc4] / 2)) * 2
            result[i, j1] = ((image[i, j2] / 2) - (image[i, nc4] / 2)) * 2
    image[:nr2, :nc2] = result[:, :]
    #第二次
    result = np.zeros((nr, nc, 3), float)
    for i in range(0, nr - 1, 2):
        for j in range(0, nc):
            i1 = (int)(i + 1)
            i2 = (int)(i / 2)
            nr4 = (int)(nr2 + i2)

            result[i, j] = ((image[i2, j] / 2) + (image[nr4, j] / 2)) * 2
            result[i1, j] = ((image[i2, j] / 2) - (image[nr4, j] / 2)) * 2

    # //copy array
    image = np.copy(result)
    # // Horizontal processing:

    for i in range(0, nr):
        for j in range(0, nc - 1, 2):
            j1 = (int)(j + 1)
            j2 = (int)(j / 2)
            nc3 = (int)(j2 + nc2)
            result[i, j] = ((image[i, j2] / 2) + (image[i, nc3] / 2)) * 2
            result[i, j1] = ((image[i, j2] / 2) - (image[i, nc3] / 2)) * 2

    resultimg = result.astype(np.uint8)

    return resultimg


if __name__ == '__main__':
    # loadImage & copy image
    image = cv2.imread("./image/lena.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image2 = waveleteTransform(image)
    image3 = inverseWaveleteTransform(image2)
    cv2.imwrite('./image/TwiceDWT.jpg', image2)
    #cv2.imwrite('./image/TwiceIDWT.jpg', image3)
    cv2.imshow('DWT',image2)
    cv2.imshow('IDWT', image3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

