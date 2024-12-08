import numpy as np
import pandas as pd
import cv2


def project_points(points, rvec, tvec, camera_matrix):
    p2D, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, None)
    return p2D


def load_data():
    cam_df = pd.read_csv(
        "data/CSV/camera_data.csv", sep=";"
    )
    dron1_df = pd.read_csv(
        "data/CSV/dron1_pos.csv", sep=";"
    )
    dron2_df = pd.read_csv(
        "data/CSV/dron2_pos.csv", sep=";"
    )
    cam_1 = cv2.VideoCapture("data/cam_1.avi")
    cam_2 = cv2.VideoCapture("data/cam_2.avi")
    cam_3 = cv2.VideoCapture("data/cam_3.avi")
    cam_4 = cv2.VideoCapture("data/cam_4.avi")
    return cam_1, cam_2, cam_3, cam_4, cam_df, dron1_df, dron2_df


def display_avi(file_avi):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while 1:
        ret, frame = file_avi.read()

        frame = cv2.resize(frame, (1200, 540))
        # applying on each frame
        fgmask = fgbg.apply(frame)

        cv2.imshow("frame", fgmask)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

    file_avi.release()
    cv2.destroyAllWindows()


def display_with_projection(file_avi, points, rvec, tvec, camera_matrix):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    scale_x = 1200 / cam_df.iloc[0, 14]
    scale_y = 540 / cam_df.iloc[0, 13]

    frame_index = 0  # Śledzenie numeru klatki
    while True:
        ret, frame = file_avi.read()
        if not ret:
            break

        # Generowanie fgmask
        fgmask = fgbg.apply(frame)

        # Resize maski
        fgmask = cv2.resize(fgmask, (1200, 540))

        # Konwersja fgmask na obraz kolorowy
        fgmask_colored = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

        # Rysowanie punktów dla bieżącej klatki
        if frame_index < len(points):  # Upewnij się, że dane punktów nie przekraczają liczby klatek
            row = points.iloc[frame_index]  # Pobierz pozycje drona dla danej klatki
            points_x_y = project_points(np.array([[row[0], row[1], row[2]]]), rvec, tvec, camera_matrix)
            x, y = int(points_x_y[0][0][0]), int(points_x_y[0][0][1])
            x = int(x * scale_x)
            y = int(y * scale_y)

            if 0 <= x < fgmask_colored.shape[1] and 0 <= y < fgmask_colored.shape[0]:
                # Zielony punkt na fgmask
                cv2.circle(fgmask_colored, (x, y), 5, (0, 255, 0), -1)

        # Wyświetlanie maski z naniesionymi punktami
        cv2.imshow("frame", fgmask_colored)

        # Obsługa zamykania okna
        if cv2.waitKey(30) & 0xFF == 27:
            break

        frame_index += 1  # Przejdź do następnej klatki

    file_avi.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cam_1, cam_2, cam_3, cam_4, cam_df, dron1_df, dron2_df = load_data()
    # display_avi(cam_1)
    print(dron1_df.head())
    focal_length = 1156.3256
    cx = cam_df.iloc[0, 14] / 2
    cy = cam_df.iloc[0, 13] / 2
    rvec = np.array(dron1_df.iloc[0, 1:4])
    tvec = np.array(dron1_df.iloc[0, 4:7])

    camera_matrix = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])

    points = dron1_df.iloc[:, 1:4]
    display_with_projection(cam_1, points, rvec, tvec, camera_matrix)