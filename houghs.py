import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from main import harris_detector


def equalize_hist(img: np.array)-> np.array:
    img = img.astype(np.uint8)
    hist = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i][j]] += 1
    hist = hist / (img.shape[0] * img.shape[1])
    hist = np.cumsum(hist)
    hist = hist * 255
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = hist[img[i, j]]
    return img


def hough(corners: list)-> np.array:
    theta = np.arange(0, 90, 1)
    rho = np.arange(0, 1000, 1)
    H = np.zeros((len(rho), len(theta)))
    for corner in corners:
        for t in theta:
            r = int(corner[0] * np.cos(np.radians(t)) + corner[1] * np.sin(np.radians(t)))
            H[r, t] += 1
    return H


def hough_peaks(H: np.array, n: int)-> list:
    peaks = []
    for i in range(n):
        maximo = np.unravel_index(np.argmax(H), H.shape)
        peaks.append(maximo)
        H[maximo] = 0
    return peaks


def hough_lines(img: np.array, peaks: list) -> None:
    output = np.zeros(img.shape)
    for peak in peaks:
        theta = peak[1]
        rho = peak[0]
        a = np.cos(np.radians(theta))
        b = np.sin(np.radians(theta))
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # Crear una imagen temporal
        temp_img = Image.new("RGB", (img.shape[1], img.shape[0]), (0, 0, 0))
        draw = ImageDraw.Draw(temp_img)

        # Dibujar la línea
        draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 255), width=2)

        # Convertir la imagen temporal a un arreglo de numpy
        temp_img = np.array(temp_img)

        # Sumar la imagen temporal a la imagen de salida
        output = np.add(output, temp_img)
    return output


def showH(H: np.array)-> None:
    plt.imshow(H, cmap='gray')
    plt.show()


def main():
    cornerlist, img = harris_detector("img.png", 3, 0.04, 0.1)
    H = hough(cornerlist)
    peaks = hough_peaks(H, 60)
    lines = hough_lines(img, peaks)
    plt.imshow(img)
    plt.imshow(lines, alpha=0.5)
    plt.show()
    H = equalize_hist(H)

    showH(H)


if __name__ == '__main__':
    main()
