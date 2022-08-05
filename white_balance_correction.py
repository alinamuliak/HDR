import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import cv2 as cv
import ffmpeg


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

def difference_plot(src, ref, src_frame=0, ref_frame=0, save=True):
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(35, 18))
    axs[0][0].imshow(cv.cvtColor(src[0:200, 0:400, :], cv.COLOR_BGR2RGB))
    axs[0][0].set_title(f'Source (frame {src_frame})', fontweight='bold')
    axs[1][0].imshow(cv.cvtColor(src, cv.COLOR_BGR2RGB))


    axs[0][3].imshow(cv.cvtColor(ref[0:200, 0:400, :], cv.COLOR_BGR2RGB))
    axs[0][3].set_title(f'Reference (frame {ref_frame})', fontweight='bold')
    axs[1][3].imshow(cv.cvtColor(ref, cv.COLOR_BGR2RGB))
    # plt.show()

    for (i, image) in enumerate((src, ref)):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        for (j, color) in enumerate('rgb'):
            hist, bins = exposure.histogram(image[..., j],
                                            source_range="dtype")
            axs[0][i + 1].plot(bins, hist / hist.max(), color=color)

            cdf, bins = exposure.cumulative_distribution(image[..., j])
            axs[0][i + 1].plot(bins, cdf, color=color)

    for (i, image) in enumerate((src, ref)):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        for (j, color) in enumerate('rgb'):
            hist, bins = exposure.histogram(image[0:400, 0:200, j],
                                            source_range="dtype")
            axs[1][i + 1].plot(bins, hist / hist.max(), color=color)

            cdf, bins = exposure.cumulative_distribution(image[0:400, 0:200, j])
            axs[1][i + 1].plot(bins, cdf, color=color)

    axs[0][1].set_title("Source", fontweight='bold')
    axs[0][2].set_title("Reference", fontweight='bold')
    # display the output plots
    plt.tight_layout()
    if save:
        plt.savefig(f'{src_frame}_{ref_frame}_hist_cdf.png')
    plt.show()




def split_video_to_frames(path):
    vidcap = cv.VideoCapture(path)
    success, first_frame = vidcap.read()
    if not success:
        return
    first_frame = cv.rotate(first_frame, cv.ROTATE_180)
    plt.imshow(first_frame[150:200, 200:300, :])
    plt.title('background piece')
    plt.show()
    # return

    counter = 1
    means = [np.mean(first_frame[150:200, 200:300, i]) for i in range(3)]
    print('means:', means)
    cv.imwrite('video/1.png', first_frame)
    counter += 1
    success, image = vidcap.read()
    image = cv.rotate(image, cv.ROTATE_180)

    while success:
        for i in range(3):
            current_mean = np.mean(image[100:200, 100:400, i])
            rescaled_image_channel = (means[i] / current_mean) * image[:, :, i]
            rescaled_image_channel[:, :][rescaled_image_channel[:, :] > 254] = 255
            image[:, :, i] = rescaled_image_channel
            # image[:, :, i] = 255 * (rescaled_image_channel - np.min(rescaled_image_channel))/np.ptp(rescaled_image_channel).astype(int)
            cv.imwrite(f'video/{counter}.png', image)
        # print('written!')
        counter += 1
        success, image = vidcap.read()
        image = cv.rotate(image, cv.ROTATE_180)
    print('done!')


if __name__ == "__main__":
    # split_video_to_frames('subject1-006-001.mp4')
    # split_video_to_frames('./current.mp4')
    vidcap = cv.VideoCapture("subject1-006-001.mp4")
    success, first = vidcap.read()
    first = cv.rotate(first, cv.ROTATE_180)
    counter = 0
    while success:
        counter += 1
        if counter == 100:
            break
        success, second = vidcap.read()
    _, second = vidcap.read()
    second = cv.rotate(second, cv.ROTATE_180)
    difference_plot(first, second, src_frame=1, ref_frame=counter, save=False)
    vidcap.release()
    cv.destroyAllWindows()
