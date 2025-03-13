import cv2
import numpy as np
import easyocr
from pathlib import Path
import os
import subprocess
import shutil


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


def process_video(video_path, languages=["en"], gpu=False, min_confidence=0.4):
    """
    Process a video file to create a text mask video.

    Args:
        video_path (str): Path to the input video
        languages (list, optional): List of languages to detect. Defaults to ['en'].
        gpu (bool, optional): Whether to use GPU. Defaults to False.
        min_confidence (float, optional): Minimum confidence threshold for detection. Defaults to 0.4.

    Returns:
        str: Path to the output mask video
    """
    # Create output directory
    video_file = Path(video_path)
    output_dir = video_file.parent / f"{video_file.stem}_output"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)
    frames_dir = output_dir / "frames"
    boxes_dir = output_dir / "boxes"
    masks_dir = output_dir / "masks"
    final_masks_dir = output_dir / "final_masks"

    os.makedirs(frames_dir)
    os.makedirs(boxes_dir)
    os.makedirs(masks_dir)
    os.makedirs(final_masks_dir)

    # First pass: count total frames in the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video at {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Count frames manually to ensure accuracy
    total_frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        total_frames += 1

    cap.release()

    duration = total_frames / fps
    print(f"Video FPS: {fps}")
    print(f"Video dimensions: {width}x{height}")
    print(f"Total frames (counted): {total_frames}")
    print(f"Duration: {duration:.2f} seconds")

    # Calculate frame processing interval (every FPS/2 frames)
    process_interval = max(1, int(fps / 2))
    print(f"Processing every {process_interval} frames")

    # Second pass: process frames
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    last_mask = None
    last_processed_frame = -process_interval  # To ensure we process the first frame

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the original frame
        frame_path = str(frames_dir / f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)

        # Check if this frame should be processed or use the previous mask
        if frame_count - last_processed_frame >= process_interval:
            # Process this frame
            print(f"Processing frame {frame_count}")
            last_processed_frame = frame_count

            # Detect text boxes
            boxes = detect_text_boxes(
                frame_path,
                languages=languages,
                gpu=gpu,
                min_confidence=min_confidence,
            )

            # Draw boxes on frame
            boxes_path = str(boxes_dir / f"frame_{frame_count:04d}_boxes.jpg")
            draw_boxes_on_image(frame_path, boxes, output_path=boxes_path)

            # Create text mask
            mask_path = str(masks_dir / f"frame_{frame_count:04d}_mask.jpg")
            mask = create_text_mask(frame_path, boxes, output_path=mask_path)
            last_mask = mask
        else:
            # Use the last processed mask
            if last_mask is not None:
                mask_path = str(final_masks_dir / f"frame_{frame_count:04d}_mask.jpg")
                cv2.imwrite(mask_path, last_mask)

        # Save the mask for this frame (either newly created or copied)
        final_mask_path = str(final_masks_dir / f"frame_{frame_count:04d}_mask.jpg")
        if last_mask is not None:
            cv2.imwrite(final_mask_path, last_mask)

        frame_count += 1

    cap.release()

    # Verify we have exactly the right number of frames
    final_mask_files = list(final_masks_dir.glob("*.jpg"))
    print(f"Generated {len(final_mask_files)} mask frames")

    if len(final_mask_files) != total_frames:
        print(
            f"WARNING: Number of mask frames ({len(final_mask_files)}) doesn't match original video ({total_frames})"
        )

    # Create the output video using ffmpeg directly with image sequence
    output_video = str(output_dir / f"{video_file.stem}_mask.mp4")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-framerate",
        str(fps),
        "-i",
        str(final_masks_dir / "frame_%04d_mask.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        output_video,
    ]

    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Mask video created at {output_video}")

    return output_video


def main():
    # Example usage with a single image
    # image_path = "4_1_first_frame.jpg"
    # boxes = detect_text_boxes(
    #     image_path,
    #     languages=["en"],
    #     gpu=False,
    #     min_confidence=0.4,
    # )
    # draw_boxes_on_image(image_path, boxes, output_path="4_1_first_frame_boxes.jpg")
    # create_text_mask(image_path, boxes, output_path="4_1_first_frame_mask.jpg")

    # Example usage with a video
    video_path = "4_3.mp4"  # Replace with your video path
    process_video(
        video_path,
        languages=["en"],
        gpu=False,
        min_confidence=0.4,
    )


if __name__ == "__main__":
    main()
