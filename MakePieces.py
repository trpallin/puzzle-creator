import cv2
import json
import numpy as np
import os
import random
import sys
import uuid
from glob import glob


def resize_image_to_fit(image, width, height):
    """Resize an image while maintaining aspect ratio."""
    h, w = image.shape[:2]
    scale = max(width / w, height / h)
    resized_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return resized_img


def apply_mask(source_img, mask_img, output_size):
    """Apply mask to source image."""
    alpha_channel_mask = mask_img[:, :, 3] / 255.0

    # Create an empty array for the output image with an alpha channel (4 channels)
    output_image = np.zeros((output_size, output_size, 4), dtype=np.uint8)

    # Multiply each color channel of the source image by the mask's alpha channel
    for c in range(3):  # Iterate over the color channels (ignoring the alpha channel)
        output_image[:, :, c] = (source_img[:, :, c] * alpha_channel_mask).astype(np.uint8)

    # Set the alpha channel of the output image to the mask's alpha channel
    output_image[:, :, 3] = (255 * alpha_channel_mask).astype(np.uint8)

    return output_image


def randomly_generate_piece_types(rows, cols):
    vertical_grids = np.zeros((rows, cols), dtype=int)
    horizontal_grids = np.zeros((rows, cols), dtype=int)
    res = np.empty((rows, cols), dtype=object)  # for storing strings

    # Populate grids with random 0 or 1
    for i in range(rows):
        for j in range(cols):
            vertical_grids[i, j] = random.randint(0, 1)
            horizontal_grids[i, j] = random.randint(0, 1)

    # Generate piece types based on the grids
    for i in range(rows):
        for j in range(cols):
            # Default values for edges
            top = bottom = left = right = 2

            if i > 0:
                top = horizontal_grids[i - 1, j]
            if i < rows - 1:
                bottom = 1 - horizontal_grids[i, j]
            if j > 0:
                left = vertical_grids[i, j - 1]
            if j < cols - 1:
                right = 1 - vertical_grids[i, j]

            # Construct the string for the puzzle piece type
            res[i, j] = f"{top}{right}{bottom}{left}"

    return res


def add_transparent_padding(img, top, right, bottom, left):
    # If the image does not have an alpha channel, add one
    if img.shape[2] < 4:
        img = np.dstack([img, np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255])

    # Calculate new image dimensions
    new_height = img.shape[0] + top + bottom
    new_width = img.shape[1] + left + right

    # Create a new image with transparent background
    padded_img = np.zeros((new_height, new_width, 4), dtype=np.uint8)

    # Copy the original image into the center of the new image
    padded_img[top:top + img.shape[0], left:left + img.shape[1]] = img

    return padded_img


def create_json_file(unique_id, name, rows, cols, file_path):
    # Data to be written to JSON
    data = {
        "uuid": unique_id,
        "name": name,
        "rows": rows,
        "cols": cols
    }

    # Writing JSON data to the specified file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"File created at {file_path}")


def create_puzzle_pieces(rows, cols, mask_img_location, img_for_puzzle, output_location):
    # Load source image
    source_img = cv2.imread(img_for_puzzle, cv2.IMREAD_UNCHANGED)
    if source_img is None:
        print(f"Error: Unable to load image {img_for_puzzle}")
        return

    # Ensure output directory exists
    os.makedirs(output_location, exist_ok=True)

    # Load all mask images
    mask_files = glob(os.path.join(mask_img_location, "*.png"))
    if not mask_files:
        print(f"Error: No mask images found in {mask_img_location}")
        return

    piece_size = 500  # Size of each puzzle piece, adjust as needed
    piece_body_size = int(piece_size * 0.54)
    piece_margin = (piece_size - piece_body_size) // 2

    # Calculate the size of the full puzzle
    full_puzzle_width = cols * piece_body_size
    full_puzzle_height = rows * piece_body_size

    # Resize the source image to fit the full puzzle size
    resized_source = resize_image_to_fit(source_img, full_puzzle_width, full_puzzle_height)
    resized_h, resized_w = resized_source.shape[:2]
    padded_source = add_transparent_padding(resized_source, piece_size, piece_size, piece_size, piece_size)
    piece_types = randomly_generate_piece_types(rows, cols)

    for row in range(rows):
        for col in range(cols):
            # Select a random mask for each piece
            mask_img_path = f"{mask_img_location}/puzzle_piece_{piece_types[row][col]}.png"
            mask_img = cv2.imread(mask_img_path, cv2.IMREAD_UNCHANGED)

            # Calculate the position of the current piece in the resized source image
            x_start = col * piece_body_size + piece_size
            y_start = row * piece_body_size + piece_size

            # Crop the current piece from the resized source image
            cropped_source = padded_source[y_start - piece_margin:y_start + piece_body_size + piece_margin,
                             x_start - piece_margin:x_start + piece_body_size + piece_margin]

            # Apply mask to the cropped source piece
            puzzle_piece = apply_mask(cropped_source, mask_img, piece_size)

            # Save the puzzle piece to the output location
            piece_filename = f"piece_{row}_{col}.png"
            cv2.imwrite(os.path.join(output_location, piece_filename), puzzle_piece)

    print(f"Puzzle pieces saved to {output_location}")


def main():
    if len(sys.argv) < 5:
        print("Not enough arguments were provided. ({puzzleName}, {imagePath}, {rows}, {cols})")
        return
    name = sys.argv[1]
    rows = int(sys.argv[3])
    cols = int(sys.argv[4])
    unique_id = str(uuid.uuid4())
    mask_img_location = "PuzzleBase/v2"
    img_for_puzzle = sys.argv[2]
    output_location = f"Result/{unique_id}/allpieces"
    json_location = f"Result/{unique_id}/puzzle_info.json"

    create_puzzle_pieces(rows, cols, mask_img_location, img_for_puzzle, output_location)
    create_json_file(unique_id, name, rows, cols, json_location)


if __name__ == "__main__":
    main()
