import cv2
import numpy as np

if __name__ == '__main__':
    # -1 -> 0.64 seconds shutter speed; -13 -> 0.00015 seconds shutter speed
    exposures = [i for i in range(-1, -14, -1)]
    suitable_exposure_index = -1
    current_exposure_index = 0  # starting from the largest (so that the first is the brightest)
    cam = cv2.VideoCapture(0)


    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cam.set(cv2.CAP_PROP_EXPOSURE, exposures[current_exposure_index])
    cv2.namedWindow("camera")
    ret, first_frame = cam.read()
    if not ret:
        print("failed to grab frame")
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    img_size = first_frame_gray.shape[0] * first_frame_gray.shape[1]

    # img_counter = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("camera", gray)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:  # SPACE pressed
            while True:
                ret, frame = cam.read()
                if not ret:
                    print("failed to grab frame")
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow("camera", gray)
                if current_exposure_index == 0:
                    if np.count_nonzero(gray >= 250) < img_size * 0.2:
                        img_name = f'underexposed{exposures[0]}.png'
                        cv2.imwrite(img_name, frame)
                        print(
                            f"THE SCENE IS TOO DARK, saved frame to {img_name} with exposure value {exposures[0]}")
                        cam.release()
                        cv2.destroyAllWindows()
                        exit()
                    cam.set(cv2.CAP_PROP_EXPOSURE, exposures[current_exposure_index])

                if np.count_nonzero(gray >= 250) > img_size * 0.2:
                    if current_exposure_index == len(exposures) - 1:
                        print(f'{np.count_nonzero(gray >= 250)} out of {img_size} are overexposed')
                        img_name = f'overexposed{exposures[current_exposure_index]}.png'
                        cv2.imwrite(img_name, frame)
                        print(f"THE SCENE IS TOO BRIGHT, saved frame to {img_name} with exposure value {exposures[current_exposure_index]}")
                        cam.release()
                        cv2.destroyAllWindows()
                        exit()
                    elif current_exposure_index < len(exposures) - 1:
                        current_exposure_index += 1
                        cam.set(cv2.CAP_PROP_EXPOSURE, exposures[current_exposure_index])

                else:
                    suitable_exposure_index = current_exposure_index
                    img_name = f'gray{exposures[current_exposure_index]}.png'
                    cv2.imwrite(img_name, frame)
                    print(f"{img_name} written!")
                    print(f"FOUNDED STARTING EXPOSURE {exposures[suitable_exposure_index]}")
                    cv2.imshow("camera", gray)
                    break
                cv2.imshow("camera", gray)

            while np.count_nonzero(gray == 255) < len(gray) * 0.2 and suitable_exposure_index > 0:
                ret, frame = cam.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if not ret:
                    print("failed to grab frame")
                    break
                cv2.imshow("camera", gray)

                suitable_exposure_index -= 1
                cam.set(cv2.CAP_PROP_EXPOSURE, exposures[current_exposure_index])
                img_name = f'gray{exposures[suitable_exposure_index]}.png'
                cv2.imwrite(img_name, frame)
                print(f"{img_name} written with {exposures[suitable_exposure_index]} exposure!")

    cam.release()
    cv2.destroyAllWindows()
