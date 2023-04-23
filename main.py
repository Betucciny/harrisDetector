import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def gauss_kernel(size: int, sigma: float):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    kernel /= 2 * np.pi * sigma ** 2
    return kernel


def conv(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    image_padded = np.zeros((image.shape[0] + kernel.shape[0] - 1, image.shape[1] + kernel.shape[1] - 1))
    image_padded[kernel.shape[0] - 2:-1, kernel.shape[1] - 2:-1] = image
    image_conv = np.zeros_like(image)
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            image_conv[y, x] = (kernel * image_padded[y:y + kernel.shape[0], x:x + kernel.shape[1]]).sum()
    return image_conv


def gradient(image):
    Ix = np.gradient(image, axis=1)
    Iy = np.gradient(image, axis=0)
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    return Ixx, Ixy, Iyy


def harris_detector(image: np.array):
    # Convert image to grayscale
    image = np.array(image.convert('L'))

    # Calculate gradients
    Ixx, Ixy, Iyy = gradient(image)

    # Calculate sums over local neighbourhoods
    Sxx = conv(Ixx, gauss_kernel(3, 1))
    Sxy = conv(Ixy, gauss_kernel(3, 1))
    Syy = conv(Iyy, gauss_kernel(3, 1))

    # Calculate corner response function
    k = 0.04
    det = (Sxx * Syy) - (Sxy ** 2)
    trace = Sxx + Syy
    R = det - k * (trace ** 2)

    # Threshold the corner points
    threshold = 0.1 * R.max()
    R[R < threshold] = 0
    corner_points = []
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i, j] > 0:
                corner_points.append([i, j])
    return corner_points


def main():
    image = Image.open('img_1.png')
    R = harris_detector(image)
    plt.imshow(image)
    plt.scatter([p[1] for p in R], [p[0] for p in R], s=1, marker='o', c='r')
    plt.title("Harris")
    plt.xticks([]), plt.yticks([])
    plt.savefig('Harris_detector', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()

