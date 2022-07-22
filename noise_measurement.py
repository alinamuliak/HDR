import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def signal_to_noise(arr, axis=None, ddof=0):
    arr = np.asanyarray(arr)
    m = arr.mean(axis)
    sd = arr.std(axis=axis, ddof=ddof)
    # return np.where(sd == 0, 0, m / sd)
    if sd == 0:
        return 0
    return m / sd


if __name__ == "__main__":
    # img = cv.imread("../data/softserve/subject1/record6")
    snr = []

    vidcap = cv.VideoCapture('../data/softserve/subject1/record6/subject1-006-001-background.mp4')
    success, image = vidcap.read()
    counter = 1
    while success:
        snr.append(signal_to_noise(image))
        if counter == 1200:
            for i, col in enumerate(['r', 'g', 'b']):
                hist = cv.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
                plt.xlim([0, 256])
            plt.title('RGB histogram', fontweight='bold')
            plt.xlabel(f'frame #{counter}')
            # plt.show()
            plt.savefig(f'plots/{counter}.png')
            plt.clf()
            break
        counter += 1
        success, image = vidcap.read()

    print('done!')

    # plt.title("SNR", fontweight="bold")
    # plt.xlabel("frame #")
    # plt.ylabel("SNR value")
    # plt.plot(snr)
    # plt.show()
    # print(snr)
