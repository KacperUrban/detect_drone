import numpy as np
import pandas as pd
import cv2

# Klasa Kamery
class Camera:
    def __init__(self, camera_id, rvec, tvec, focal_length, cx, cy):
        self.camera_id = camera_id
        self.rvec = rvec
        self.tvec = tvec
        self.camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])
        self.upVec = np.array([0.0, 0.0, -1.0])  # Wektor skierowany do góry

    def project_points(self, points):
        p2D, _ = cv2.projectPoints(points, self.rvec, self.tvec, self.camera_matrix, None)
        return p2D

# Klasa Drona
class Drone:
    def __init__(self, drone_id, position_data, height, length):
        self.drone_id = drone_id
        self.position_data = position_data
        self.current_index = 0
        self.height = height
        self.length = length

    def update_position(self):
        if self.current_index < len(self.position_data) - 1:
            self.current_index += 1
            return np.array([
                self.position_data.iloc[self.current_index, 1],  # x
                self.position_data.iloc[self.current_index, 2],  # y
                self.position_data.iloc[self.current_index, 3]   # z
            ], np.float32)
        return None

def display_with_projection(cameras, drones, video_files):
    fgbgs = [cv2.createBackgroundSubtractorMOG2() for _ in video_files]

    while True:
        frames = []
        for file, fgbg in zip(video_files, fgbgs):
            ret, frame = file.read()
            if not ret:
                print("End of video stream.")
                return
            fgmask = fgbg.apply(frame)
            fgmask_resized = cv2.resize(fgmask, (600, 400))  # Skalowanie obrazu
            frames.append(fgmask_resized)

        processed_frames = []
        for fgmask_resized, camera, drone in zip(frames, cameras, drones):
            # Aktualizuj pozycję drona
            drone_position = drone.update_position()
            if drone_position is None:
                print(f"No more positions for Drone {drone.drone_id}.")
                continue

            # Debug: Wyświetlenie pozycji drona w 3D
            print(f"Drone {drone.drone_id} 3D position: {drone_position}")

            # Rzutowanie pozycji drona
            projected_point, _ = cv2.projectPoints(np.array([drone_position]), camera.rvec, camera.tvec, camera.camera_matrix, None)
            x, y = int(projected_point[0][0][0]), int(projected_point[0][0][1])

            # Skaluje współrzędne punktu w zależności od rozmiaru obrazu
            height, width = fgmask_resized.shape
            scale_x = width / float(frame.shape[1])
            scale_y = height / float(frame.shape[0])
            x_rescaled = int(x * scale_x) // 10
            y_rescaled = int(y * scale_y) // 10

            print(f"Drone {drone.drone_id} projected point: (x={x_rescaled}, y={y_rescaled})")

            # Nanoszenie punktu na obraz
            fgmask_colored = cv2.cvtColor(fgmask_resized, cv2.COLOR_GRAY2BGR)
            if 0 <= x_rescaled < fgmask_colored.shape[1] and 0 <= y_rescaled < fgmask_colored.shape[0]:
                cv2.circle(fgmask_colored, (x_rescaled, y_rescaled), 5, (0, 255, 0), -1)  # Zielony punkt
                cv2.putText(fgmask_colored, f"Drone {drone.drone_id}", (x_rescaled + 10, y_rescaled - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            else:
                print(f"Projected point for Drone {drone.drone_id} is out of frame bounds!")

            processed_frames.append(fgmask_colored)

        # Wyświetlanie wszystkich klatek w jednym oknie
        if processed_frames:
            combined_frame = np.hstack(processed_frames)
            cv2.imshow("Drones Tracking", combined_frame)

        if cv2.waitKey(30) & 0xFF == 27:  # Klawisz ESC
            break

    for file in video_files:
        file.release()
    cv2.destroyAllWindows()


# Główna część programu
if __name__ == "__main__":
    # Wczytanie danych
    cam_df = pd.read_csv("data/CSV/camera_data.csv", sep=";")
    dron1_df = pd.read_csv("data/CSV/dron1_pos.csv", sep=";")
    dron2_df = pd.read_csv("data/CSV/dron2_pos.csv", sep=";")
    video_files = [
        cv2.VideoCapture("data/cam_1.avi"),
        cv2.VideoCapture("data/cam_2.avi"),
        cv2.VideoCapture("data/cam_3.avi"),
        cv2.VideoCapture("data/cam_4.avi")
    ]

    # Tworzenie obiektów kamer
    cameras = []
    focal_length = 1156.3256
    for i in range(4):
        cx = cam_df.iloc[i, 14] / 2
        cy = cam_df.iloc[i, 13] / 2
        rvec = np.array([cam_df.iloc[i, 1], cam_df.iloc[i, 2], cam_df.iloc[i, 3]])
        tvec = np.array([cam_df.iloc[i, 4], cam_df.iloc[i, 5], cam_df.iloc[i, 6]])
        cameras.append(Camera(i, rvec, tvec, focal_length, cx, cy))

    # Tworzenie obiektów dronów
    drones = [
        Drone(1, dron1_df, height=0.15, length=0.35),
        Drone(2, dron2_df, height=0.15, length=0.35)
    ]

    # Uruchomienie przetwarzania
    display_with_projection(cameras, drones, video_files)
