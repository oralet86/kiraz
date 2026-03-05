import cv2
import os
import numpy as np
from typing import List
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm


def generate_clean_aruco_pdf(
    filename: str, marker_ids: list[int], marker_size_cm: float
) -> None:
    # 1. Setup ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    tmp_dir = "temp_markers"
    os.makedirs(tmp_dir, exist_ok=True)

    marker_files = []
    for m_id in marker_ids:
        img = cv2.aruco.generateImageMarker(aruco_dict, m_id, 800)
        path = os.path.join(tmp_dir, f"marker_{m_id}.png")
        cv2.imwrite(path, img)
        marker_files.append(path)

    # 2. Create PDF
    c = canvas.Canvas(filename, pagesize=A4)
    page_width, page_height = A4

    # Define the physical size using the cm constant
    size = marker_size_cm * cm
    margin = 1 * cm  # Distance from the edge of the paper

    # Coordinate mapping (x, y)
    # Note: (0,0) is the bottom-left corner of the page
    positions = [
        (margin, page_height - margin - size),  # Top Left
        (page_width - margin - size, page_height - margin - size),  # Top Right
        (margin, margin),  # Bottom Left
        (page_width - margin - size, margin),  # Bottom Right
    ]

    for i, (x, y) in enumerate(positions):
        if i < len(marker_files):
            # drawImage(image_path, x_pos, y_pos, width, height)
            c.drawImage(marker_files[i], x, y, width=size, height=size)

    c.save()
    print(f"Success: '{filename}' created with {marker_size_cm}cm markers.")

    # 3. Cleanup
    for f in marker_files:
        os.remove(f)
    os.rmdir(tmp_dir)


def start_live_demo(target_ids: List[int], marker_size_cm: float = 5.0) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    cap = cv2.VideoCapture(0)

    # Placeholder Calibration (Required for drawFrameAxes)
    # Once you calibrate, replace these with your actual camera matrix
    ret, frame = cap.read()
    if not ret:
        return
    h, w = frame.shape[:2]
    focal_length = w  # Approximation
    cam_matrix = np.array(
        [[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion for now

    # Coordinates of a square marker in its own 3D space
    obj_points = np.array(
        [
            [-marker_size_cm / 2, marker_size_cm / 2, 0],
            [marker_size_cm / 2, marker_size_cm / 2, 0],
            [marker_size_cm / 2, -marker_size_cm / 2, 0],
            [-marker_size_cm / 2, -marker_size_cm / 2, 0],
        ],
        dtype=np.float32,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            for i in range(len(ids)):
                m_id = int(ids[i][0])
                if m_id in target_ids:
                    # 1. Draw the 2D marker border
                    cv2.aruco.drawDetectedMarkers(frame, [corners[i]], ids[i])

                    # 2. Estimate Pose (SolvePnP)
                    # Returns rotation (rvec) and translation (tvec) vectors
                    success, rvec, tvec = cv2.solvePnP(
                        obj_points, corners[i], cam_matrix, dist_coeffs
                    )

                    if success:
                        # 3. Draw 3D Axis (Red=X, Green=Y, Blue=Z)
                        # The last parameter (3.0) is the length of the axis in cm
                        cv2.drawFrameAxes(
                            frame, cam_matrix, dist_coeffs, rvec, tvec, 3.0
                        )

        cv2.imshow("ArUco 3D Axis Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def generate_chessboard_pdf(
    filename: str, square_size_cm: float, rows: int, cols: int
) -> None:
    """
    Generates a chessboard pattern PDF.
    rows/cols: number of squares (e.g., 7x10).
    """
    c = canvas.Canvas(filename, pagesize=A4)
    page_width, page_height = A4

    # Calculate total grid size
    grid_width = cols * square_size_cm * cm
    grid_height = rows * square_size_cm * cm

    # Center the grid on the A4 page
    start_x = (page_width - grid_width) / 2
    start_y = (page_height - grid_height) / 2

    for r in range(rows):
        for col in range(cols):
            # Alternate black and white squares
            if (r + col) % 2 == 0:
                c.setFillColorRGB(0, 0, 0)  # Black
                c.rect(
                    start_x + (col * square_size_cm * cm),
                    start_y + (r * square_size_cm * cm),
                    square_size_cm * cm,
                    square_size_cm * cm,
                    fill=1,
                )

    c.save()
    print(f"Success: '{filename}' created.")
    print(f"Grid: {cols}x{rows} squares, each {square_size_cm}cm.")


if __name__ == "__main__":
    # Specify your ID list and the desired physical size in cm here
    MARKER_IDS = [0, 1, 2, 3]

    # generate_clean_aruco_pdf(filename="aruco_sheet.pdf", marker_ids=MARKER_IDS, marker_size_cm=5.0)
    # start_live_demo(target_ids=MARKER_IDS)

    generate_chessboard_pdf(
        filename="chessboard.pdf", square_size_cm=2.5, rows=10, cols=7
    )
