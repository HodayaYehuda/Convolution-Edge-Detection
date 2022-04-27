import math
from math import *
import numpy as np
import cv2


# Hodaya Yehuda 318925617


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """

    k = k_size[::-1]
    a = np.pad(in_signal, (len(k) - 1, len(k) - 1), 'constant')
    res = np.zeros(len(a) - len(k) + 1)

    for i in range(0, len(a) - len(k) + 1):
        res[i] = np.multiply(a[i:i + len(k)], k).sum()

    return res


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """

    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros_like(in_image)

    # padded_image = np.zeros((in_image.shape[0] + len(kernel) + 1, in_image.shape[1] + len(kernel[0]) + 1))
    ker_x = int(kernel.shape[0] / 2)
    ker_y = int(kernel.shape[1] / 2)
    padded_image = np.pad(in_image, (ker_x, ker_y), 'edge')

    # padded_image[len(kernel) - 2:-(len(kernel) - 2), len(kernel[0]) - 2:-(len(kernel[0]) - 2)] = in_image
    # padded_image = np.pad(in_image, [(2,), (2,)], mode='constant')

    for x in range(in_image.shape[1]):
        for y in range(in_image.shape[0]):
            output[y, x] = (kernel * padded_image[y: y + (len(kernel[0])), x: x + (len(kernel[1]))]).sum()

    return output


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    x_array = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
    y_array = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])

    Derivative_x = cv2.filter2D(in_image, -1, x_array, borderType=1)
    Derivative_y = cv2.filter2D(in_image, -1, y_array, borderType=1)

    Mag = (((Derivative_x ** 2) + (Derivative_y ** 2)) ** 0.5)

    Direction = np.arctan(np.divide(Derivative_y, Derivative_x + 0.000001)).astype(int)

    return Direction, Mag


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    gaussian_im = np.zeros(shape=(k_size, k_size))
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8

    for x in range(0, k_size):
        for y in range(0, k_size):
            gaussian_im[x, y] = math.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) / (math.pi * (sigma ** 2) * 2)
    return conv2D(in_image, gaussian_im)


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    size = (k_size, k_size)
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    ans = cv2.GaussianBlur(in_image, size, sigma)

    return ans


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """

    # img = img.astype(float)
    kernal = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    flip_kernel = np.flip(kernal)
    filter_img = cv2.filter2D(img, -1, flip_kernel, borderType=cv2.BORDER_REPLICATE)
    new_img = np.zeros_like(filter_img)

    for i in range(1, filter_img.shape[0] - 1):
        for j in range(1, filter_img.shape[1] - 1):
            if filter_img[i][j - 1] > 0 and filter_img[i][j + 1] < 0 and filter_img[i][j - 1] > filter_img[i][j] > \
                    filter_img[i][j + 1]:
                new_img[i][j] = 1
    return new_img


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """

    return


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    myImg = (img * 255).astype(np.uint8)
    canny_Img = cv2.Canny(myImg, 100, 200)

    rows, cols = canny_Img.shape

    edges = []
    points = []
    circles = []
    num_of_show = {}

    steps = 100

    # take the limits from the canny function
    for x in range(rows):
        for y in range(cols):
            if canny_Img[x, y] == 255:
                edges.append((x, y))

    # all the possible points
    for radius in range(min_radius, max_radius + 1):
        for s in range(100):  # 100 steps
            x = int(radius * cos(2 * pi * s / 100))
            y = int(radius * sin(2 * pi * s / 100))
            points.append((x, y, radius))

    for x1, y1 in edges:
        for x2, y2, radius in points:
            new_y = y1 - y2
            new_x = x1 - x2
            count = num_of_show.get((new_y, new_x, radius))
            if count is None:
                count = 0
            num_of_show[(new_y, new_x, radius)] = count + 1

    sortedCircles = sorted(num_of_show.items(), key=lambda i: -i[1])  # sorted the array by x value

    for circle, counter in sortedCircles:
        my_x, my_y, my_radios = circle
        if counter / steps >= 0.46:  # thresh hold
            if all((my_x - sec_x) ** 2 + (my_y - sec_y) ** 2 > sec_radios ** 2 for sec_x, sec_y, sec_radios in circles):
                circles.append((my_x, my_y, my_radios))  # add circle to the final answer

    return circles


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    # cv2 answer for comparing
    filtered_image_OpenCV = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)

    filtered_image = np.zeros(in_image.shape)  # my output image

    i = 0
    while i < len(in_image):
        j = 0
        while j < len(in_image[0]):
            # for every filtered_image[i,j] we calculate the new val
            help_bilateral(in_image, filtered_image, i, j, k_size, sigma_color, sigma_space)
            j += 1
        i += 1

    return filtered_image_OpenCV, filtered_image


def help_bilateral(in_image, filtered_image, x, y, diameter, sigma_i, sigma_s):
    new_diameter = diameter / 2
    filtered_sum = 0
    sum_of_gaussian = 0

    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = int(x - (new_diameter - i))
            neighbour_y = int(y - (new_diameter - j))

            if neighbour_x >= len(in_image):
                neighbour_x = len(in_image) - 1 
            if neighbour_x < 0:
                neighbour_x = 0
            if neighbour_y >= len(in_image[0]):
                neighbour_y = len(in_image[0]) - 1
            if neighbour_y < 0:
                neighbour_y = 0

            gaussian_1 = gaussian(in_image[neighbour_x][neighbour_y] - in_image[x][y], sigma_i)
            gaussian_2 = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)

            new_gaussian = gaussian_1 * gaussian_2
            filtered_sum += in_image[neighbour_x][neighbour_y] * new_gaussian
            sum_of_gaussian += new_gaussian
            j += 1
        i += 1

    filtered_sum = filtered_sum / sum_of_gaussian  # avg
    filtered_image[x][y] = int(round(filtered_sum))


def distance(x, y, i, j):
    return np.sqrt((x - i) ** 2 + (y - j) ** 2)


def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))
