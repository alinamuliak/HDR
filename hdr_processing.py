import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def calculate_response(images: list, exposures: list):
    calibration = cv.createCalibrateDebevec()
    return calibration.process(images, exposures)


def plot_response_curves(response: list, channel: str, log=False):
    if log:
        plt.semilogy( [i for i in range(256)], np.reshape(response, 256), c=channel)  # log
        plt.ylabel('log exposure g(z)')
    else:
        plt.plot([i for i in range(256)], np.reshape(response, 256), c=channel)  # simple
        plt.ylabel('exposure g(z)')

    plt.xlabel('pixel value z')
    plt.title('camera response function', weight='bold')
    # plt.fill_between(
    #     x=[i for i in range(256)],
    #     y1=np.reshape(response, 256),
    #     color=channel,
    #     alpha=0.2)


def merge_hdr(images, output_filename="merged_hdr.png"):
    merge_mertens = cv.createMergeMertens()
    fusion = merge_mertens.process(images)

    cv.imwrite(output_filename, fusion * 255)  # hdr image
    image = cv.imread(output_filename)
    cv.namedWindow("images")
    cv.imshow("images", image)


if __name__ == "__main__":
    exposures = [-i for i in range(7, 10)]
    files = [f'gray-{i}.png' for i in range(7, 10)]
    read_images = list([cv.imread(f) for f in files])
    # exposure times in seconds
    exposure_value_in_sec = {-1: 0.64, -2: 0.32, -3: 0.16, -4: 0.08,
                             -5: 0.04, -6: 0.02, -7: 0.01, -8: 0.005,
                             -9: 0.0025, -10: 0.00125, -11: 0.00065,
                             -12: 0.000312, -13: 0.00015}
    exposures = np.float32([exposure_value_in_sec[e] for e in exposures])
    response = calculate_response(read_images, exposures)

    rgb = ['r', 'g', 'b']
    for i in range(len(rgb)):
        plot_response_curves(response[:, :, i], rgb[i], log=False)
    plt.show()
    merge_hdr(read_images)
