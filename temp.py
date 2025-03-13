import cv2
import numpy as np
import easyocr
from pathlib import Path


def detect_text_boxes(image_path, languages=["en"], gpu=False, min_confidence=0.4):
    """
    Detect text in an image using EasyOCR and return bounding boxes coordinates.

    Args:
        image_path (str): Path to the input image
        languages (list, optional): List of languages to detect. Defaults to ['en'].
        gpu (bool, optional): Whether to use GPU. Defaults to False.
        min_confidence (float, optional): Minimum confidence threshold for detection. Defaults to 0.4.

    Returns:
        list: List of tuples (bbox, text, confidence) where bbox is a list of 4 points
    """
    # Initialize the OCR reader
    reader = easyocr.Reader(languages, gpu=gpu)

    # Detect text with optimized parameters
    results = reader.readtext(
        image_path,
        detail=1,
        paragraph=False,
        width_ths=0.5,
        add_margin=0.05,
        text_threshold=0.7,
        link_threshold=0.2,
        mag_ratio=1.5,
    )

    # Filter results by confidence
    results = [r for r in results if r[2] >= min_confidence]

    # Process bounding boxes - shrink by 5%
    processed_results = []
    for bbox, text, prob in results:
        # Convert bbox to numpy array
        points = np.array(bbox, dtype=np.float32)

        # Calculate the center of the bounding box
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])

        # Shrink the bounding box by 5% towards its center
        shrink_factor = 0.95  # 5% reduction
        for i in range(len(points)):
            # Move each point 5% closer to the center
            points[i, 0] = (points[i, 0] - center_x) * shrink_factor + center_x
            points[i, 1] = (points[i, 1] - center_y) * shrink_factor + center_y

        processed_results.append((points.tolist(), text, prob))

    return processed_results


def draw_boxes_on_image(image_path, boxes, output_path=None):
    """
    Draw bounding boxes on the original image without annotations.

    Args:
        image_path (str): Path to the input image
        boxes (list): List of bounding boxes from detect_text_boxes
        output_path (str, optional): Path to save the output image. If None, will use input_name_boxes.jpg

    Returns:
        numpy.ndarray: Image with drawn boxes
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Make a copy for drawing
    result_image = image.copy()

    # Draw boxes
    for bbox, _, _ in boxes:
        # Convert to integer points for drawing
        points = np.array(bbox, dtype=np.int32)

        # Draw the bounding box
        cv2.polylines(
            result_image, [points], isClosed=True, color=(0, 255, 0), thickness=2
        )

    # Save the result if output_path is provided
    if output_path is None:
        input_path = Path(image_path)
        output_path = str(
            input_path.parent / f"{input_path.stem}_boxes{input_path.suffix}"
        )

    cv2.imwrite(output_path, result_image)
    print(f"Image with boxes saved to {output_path}")

    return result_image


def create_text_mask(image_path, boxes, output_path=None):
    """
    Create a black and white mask where text boxes are white on a black background.

    Args:
        image_path (str): Path to the input image (needed for dimensions)
        boxes (list): List of bounding boxes from detect_text_boxes
        output_path (str, optional): Path to save the output mask. If None, will use input_name_mask.jpg

    Returns:
        numpy.ndarray: Black and white mask
    """
    # Read the image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Create a black image of the same size
    height, width = image.shape[:2]
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw white filled boxes
    for bbox, _, _ in boxes:
        # Convert to integer points for drawing
        points = np.array(bbox, dtype=np.int32)

        # Fill the bounding box with white
        cv2.fillPoly(mask, [points], color=(255, 255, 255))

    # Save the result if output_path is provided
    if output_path is None:
        input_path = Path(image_path)
        output_path = str(
            input_path.parent / f"{input_path.stem}_mask{input_path.suffix}"
        )

    cv2.imwrite(output_path, mask)
    print(f"Mask saved to {output_path}")

    return mask


def main():
    image_path = "4_1_first_frame.jpg"

    # Detect text boxes
    boxes = detect_text_boxes(
        image_path,
        languages=["en"],
        gpu=False,
        min_confidence=0.4,
    )

    # Draw boxes on original image
    draw_boxes_on_image(image_path, boxes, output_path="4_1_first_frame_boxes.jpg")

    # Create text mask
    create_text_mask(image_path, boxes, output_path="4_1_first_frame_mask.jpg")


if __name__ == "__main__":
    main()
