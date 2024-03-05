import os
import time
import glob
from PIL import Image


def convert():
    # Get a list of all .webp files in the directory
    webp_files = glob.glob(os.path.join(webp_path, '*.webp'))

    for webp_image_path in webp_files:
        print(f"Processing {webp_image_path} ...")

        # Load the .webp image
        image = Image.open(webp_image_path).convert("RGB")

        current_time_millis = int(round(time.time() * 1000))
        time.sleep(0.001)

        jpg_image_name = f"img_{current_time_millis}.jpg"
        jpg_image_path = os.path.join(output_path, jpg_image_name)

        image.save(jpg_image_path, "jpeg")

        os.remove(webp_image_path)

        print(f"Converted {webp_image_path} and saved as {jpg_image_path}")


if __name__ == '__main__':
    webp_path = './NotProcessed'
    output_path = './Images'
    convert()
