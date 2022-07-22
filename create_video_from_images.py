import cv2


def create_video(img_names: list):
    img1 = cv2.imread(img_names[0])
    w, h, _ = img1.shape

    video = cv2.VideoWriter('video.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            15, (w, h))

    for img in img_names:
        frame = cv2.imread(img)
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    images = [f'plots/{i}.png' for i in range(1, 1798)]
    create_video(images)
