import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import cv2 as cv


def match_histograms(src, ref):
    # src = cv.imread(src_name)
    # ref = cv.imread(ref_name)
    matched = exposure.match_histograms(src, ref, channel_axis=-1)
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 8))
    axs[0].imshow(cv.cvtColor(src, cv.COLOR_BGR2RGB))
    axs[0].set_title('Source', fontweight='bold')
    axs[1].imshow(cv.cvtColor(ref, cv.COLOR_BGR2RGB))
    axs[1].set_title('Reference', fontweight='bold')
    axs[2].imshow(cv.cvtColor(matched, cv.COLOR_BGR2RGB))
    axs[2].set_title('Matched', fontweight='bold')
    plt.show()

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
    for (i, image) in enumerate((src, ref, matched)):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        for (j, color) in enumerate('rgb'):
            hist, bins = exposure.histogram(image[..., j],
                                            source_range="dtype")
            axs[i].plot(bins, hist / hist.max(), color=color)

            # (cdf, bins) = exposure.cumulative_distribution(image[..., j])
            # axs[i].plot(bins, cdf, color=color)

    axs[0].set_title("Source", fontweight='bold')
    axs[1].set_title("Reference", fontweight='bold')
    axs[2].set_title("Matched", fontweight='bold')
    # display the output plots
    plt.tight_layout()
    plt.show()


def split_video_to_frames(path):
    vidcap = cv.VideoCapture(path)
    success, first_frame = vidcap.read()
    if not success:
        return

    counter = 1
    means = [np.mean(first_frame[0:400, 0:200, i]) for i in range(3)]
    print('means:', means)
    cv.imwrite('video/1.png', first_frame)
    counter += 1
    success, image = vidcap.read()
    while success:
        for i in range(3):
            current_mean = np.mean(image[0:400, 0:200, i])
            rescaled_image_channel = (means[i] / current_mean) * image[:, :, i]
            image[:, :, i] = 255 * (rescaled_image_channel - np.min(rescaled_image_channel))/np.ptp(rescaled_image_channel).astype(int)
            cv.imwrite(f'video/{counter}.png', image)
        # print('written!')
        counter += 1
        success, image = vidcap.read()
    print('done!')


if __name__ == "__main__":
    split_video_to_frames('../data/softserve/subject1/record6/subject1-006-001.mp4')
    # split_video_to_frames('./current.mp4')
