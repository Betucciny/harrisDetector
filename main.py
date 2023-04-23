import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def harris_detector(img_dir, window_size, k, threshold):
    img = Image.open(img_dir)
    img = np.array(img)
    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])

    height = img.shape[0]
    width = img.shape[1]
    matrix_R = np.zeros((height, width))
    dy, dx = np.gradient(gray)

    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxy = dx * dy

    offset = int(window_size / 2)
    print("Finding Corners...")
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            Sx2 = np.sum(dx2[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sy2 = np.sum(dy2[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sxy = np.sum(dxy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            H = np.array([[Sx2, Sxy], [Sxy, Sy2]])
            det = np.linalg.det(H)
            tr = np.matrix.trace(H)
            R = det - k * (tr ** 2)
            matrix_R[y - offset, x - offset] = R

    cornerList = []
    matrix_R = matrix_R / np.max(matrix_R)
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            value = matrix_R[y, x]
            if value > threshold:
                cornerList.append([x, y, value])

    return cornerList, img


def show_corners(img_dir):
    corner_list, img = harris_detector(img_dir, 3, 0.04, 0.1)
    img = np.array(img)
    plt.imshow(img)
    plt.scatter([p[0] for p in corner_list], [p[1] for p in corner_list], s=1, marker='o', c='r')
    plt.show()


def main():
    show_corners("img.png")
    show_corners("img_1.jpg")


if __name__ == '__main__':
    main()

