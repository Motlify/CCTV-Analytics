import cv2
import numpy as np
import json
import pathlib
import logging
from utils.config import configure_logging

configure_logging()
# Initialize global variables
current_polygon = []
drawing = False
polygons = []


def draw_polygon(event, x, y, flags, param):
    global current_polygon, drawing, img_copy, img

    if event == cv2.EVENT_LBUTTONDOWN:
        # Add point to the current polygon
        current_polygon.append((x, y))
        drawing = True

        # Draw the current polygon as it is being constructed
        if len(current_polygon) > 1:
            cv2.polylines(img_copy, [np.array(current_polygon)], False, (0, 255, 0), 2)

        # Draw lines connecting the last point to the cursor
        cv2.imshow("image", img_copy)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Draw polygon as user moves the mouse
        img_copy = img.copy()
        if len(current_polygon) > 1:
            cv2.polylines(img_copy, [np.array(current_polygon)], False, (0, 255, 0), 2)
        cv2.line(img_copy, current_polygon[-1], (x, y), (0, 255, 0), 2)
        cv2.imshow("image", img_copy)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_polygon) > 2:
            # Complete the polygon by connecting the last point to the first point
            cv2.polylines(img, [np.array(current_polygon)], True, (0, 255, 0), 2)

            # Get the name for the polygon from the user
            polygon_name = input("Enter a name for the polygon: ")

            # Append polygon and name to the list
            polygons.append({"name": polygon_name, "polygon": current_polygon.copy()})

            # Reset current polygon
            current_polygon = []
        drawing = False


def run():
    global img, img_copy, current_polygon, polygons

    # Initialize polygons list for each image
    polygons = []

    for img_file_path in pathlib.Path("images").glob("*.jpg"):
        logging.info(img_file_path.absolute())

        # Load the image
        img = cv2.imread(str(img_file_path.absolute()))

        if img is None:
            logging.error(f"Failed to load image: {str(img_file_path.absolute())}")
            continue  # Skip this file and move to the next one

        img_copy = img.copy()

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", draw_polygon)

        while True:
            cv2.imshow("image", img_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                # Clear current polygon if the user wants to reset it
                current_polygon = []
                img_copy = img.copy()

        # Save the polygons with names to a JSON file
        polygon_file = f"{img_file_path.with_suffix('.json')}"
        if len(polygons) > 0:
            with open(polygon_file, "w") as f:
                json.dump(polygons, f, indent=4)
        polygons = []
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
