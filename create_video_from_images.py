import cv2


def create_video(img_names: list):
    img1 = cv2.imread(img_names[0])
    w, h, _ = img1.shape

    video = cv2.VideoWriter('video.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            30, (w, h))

    for img in img_names:
        frame = cv2.imread(img)
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    images = [f'video/{i}.png' for i in range(1, 1802)]
    img1 = cv2.imread('video/1.png')
    w, h, _ = img1.shape
    video = cv2.VideoWriter('video.avi',
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            30, (h, w))

    # vid_capture = cv2.VideoCapture('video/%01d.png')
    # output = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w, h))

    for image in images:
        frame = cv2.imread(image)
        video.write(frame)
    video.release()
    cv2.destroyAllWindows()

