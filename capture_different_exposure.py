import cv2
from time import sleep

if __name__ == '__main__':
    exposures = [-13, -6, -1]
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_EXPOSURE, exposures[0])

    cv2.namedWindow("camera")

    img_counter = 0
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("camera", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            for i in range(len(exposures)):
                cam.set(cv2.CAP_PROP_EXPOSURE, exposures[i])
                sleep(1)
                ret, frame = cam.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if not ret:
                    print("failed to grab frame")
                    break
                cv2.imshow("camera", frame)

                img_name = f'opencv_frame_{img_counter}.png'
                cv2.imwrite(img_name, frame)
                print(f"{img_name} written!")
                img_counter += 1
            cam.set(cv2.CAP_PROP_EXPOSURE, exposures[0])

    cam.release()
    cv2.destroyAllWindows()
