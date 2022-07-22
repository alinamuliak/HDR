from skimage import exposure
import matplotlib.pyplot as plt
import cv2


def match_histograms(src_name, ref_name):
    src = cv2.imread(src_name)
    ref = cv2.imread(ref_name)

    multi = True if src.shape[-1] > 1 else False
    matched = exposure.match_histograms(src, ref, multichannel=multi)
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 8))
    axs[0].imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Source', fontweight='bold')
    axs[1].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Reference', fontweight='bold')
    axs[2].imshow(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
    axs[2].set_title('Matched', fontweight='bold')
    plt.show()

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
    for (i, image) in enumerate((src, ref, matched)):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
    plt.title('RGB histograms', fontweight='bold')
    plt.show()


if __name__ == "__main__":
    match_histograms('data/example2.jpg', 'data/example1.jpg')
