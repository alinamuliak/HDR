import cv2
from time import sleep

import numpy as np

if __name__ == '__main__':
    # -1 -> 0.64 seconds shutter speed; -13 -> 0.00015 seconds shutter speed
    exposures = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13]
    suitable_exposure_index = -1
    current_exposure_index = 0  # starting from the largest (so that the first is the brightest)
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_EXPOSURE, exposures[current_exposure_index])
    cv2.namedWindow("camera")

    img_counter = 0
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("camera", gray)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:  # SPACE pressed
            while True:
                sleep(1)
                ret, frame = cam.read()
                if not ret:
                    print("failed to grab frame")
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow("camera", gray)
                if np.amax(gray) >= 255:

                    if current_exposure_index == len(exposures) - 1:
                        img_name = f'overexposed.png'
                        cv2.imwrite(img_name, gray)
                        print(f"THE SCENE IS TOO BRIGHT, saved frame to {img_name} with exposure value {exposures[current_exposure_index]}")
                        cam.release()
                        cv2.destroyAllWindows()
                        exit()
                    elif current_exposure_index < len(exposures) - 1:
                        current_exposure_index += 1
                        cam.set(cv2.CAP_PROP_EXPOSURE, exposures[current_exposure_index])

                elif np.amax(gray) < 1 and current_exposure_index == 0:
                    img_name = f'underexposed.png'
                    cv2.imwrite(img_name, gray)
                    print(f"THE SCENE IS TOO DARK, saved frame to {img_name} with exposure value {exposures[current_exposure_index]}")
                    cam.release()
                    cv2.destroyAllWindows()
                    exit()

                else:
                    cam.set(cv2.CAP_PROP_EXPOSURE, exposures[0])
                    ret, frame = cam.read()
                    if not ret:
                        print("failed to grab frame")
                        break
                    gray_smallest_shutter_speed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if np.count_nonzero(gray_smallest_shutter_speed >= 255) < len(gray_smallest_shutter_speed) * 0.2:
                        img_name = f'underexposed.png'
                        cv2.imwrite(img_name, gray)
                        print(
                            f"THE SCENE IS TOO DARK, saved frame to {img_name} with exposure value {exposures[0]}")
                        cam.release()
                        cv2.destroyAllWindows()
                        exit()
                    cam.set(cv2.CAP_PROP_EXPOSURE, exposures[current_exposure_index])
                    suitable_exposure_index = current_exposure_index
                    img_name = f'gray_{img_counter}.png'
                    img_counter += 1
                    cv2.imwrite(img_name, gray)
                    print(f"{img_name} written!")
                    print(f"FOUNDED STARTING EXPOSURE {exposures[suitable_exposure_index]}")
                    break
                cv2.imshow("camera", gray)

            while np.count_nonzero(gray == 255) < len(gray) * 0.2 and suitable_exposure_index >= 0:
                suitable_exposure_index -= 1
                cam.set(cv2.CAP_PROP_EXPOSURE, exposures[current_exposure_index])
                img_name = f'gray_{img_counter}.png'
                img_counter += 1
                cv2.imwrite(img_name, gray)
                print(f"{img_name} written with {exposures[suitable_exposure_index]} exposure!")

    cam.release()
    cv2.destroyAllWindows()
