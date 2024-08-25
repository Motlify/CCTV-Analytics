import logging
import pathlib
import json
import traceback
from PIL import Image, ImageDraw, ImageFont

from cctv_analytics.utils.config import configure_logging

configure_logging()


def draw_polygons(image, preds, label=True):
    # Convert image to RGBA to allow transparency
    image = image.convert("RGBA")

    # Create an overlay image for the semi-transparent polygon
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Font settings
    font = ImageFont.load_default()

    # Iterate through each polygon in the predictions
    for item in preds:
        name = item["name"]
        polygon = item["polygon"]

        # Ensure the polygon coordinates are tuples
        polygon = [tuple(coord) for coord in polygon]

        # Define the semi-transparent color (red with alpha 0.3)
        fill_color = (255, 0, 0, int(255 * 0.3))  # Red with alpha 0.3

        # Draw the filled polygon on the overlay
        overlay_draw.polygon(polygon, fill=fill_color, outline="red")

        if label:
            # Define the label position
            label_position = polygon[0]

            # Get the bounding box of the text
            bbox = overlay_draw.textbbox(label_position, name, font=font)

            # Draw the background rectangle (black with full opacity)
            overlay_draw.rectangle(bbox, fill=(0, 0, 0, 255))

            # Draw the label text on top of the background rectangle
            overlay_draw.text(
                (label_position[0], label_position[1]),
                name,
                fill=(255, 255, 255, 255),
                font=font,
            )

    # Composite the overlay with the original image
    combined = Image.alpha_composite(image, overlay)

    # Show the image
    combined.show()


def run():
    for img_file_path in pathlib.Path("images").glob("*.jpg"):
        image = Image.open(img_file_path.absolute())
        try:
            data = json.load(open(img_file_path.with_suffix(".json")))
            draw_polygons(image, data, True)
        except FileNotFoundError as e:
            logging.error("No file fo camera " + img_file_path.stem)
