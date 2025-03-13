import cv2
import numpy as np
import easyocr
from pathlib import Path
import os
import subprocess
import shutil


def detect_text_boxes(image_path, languages=["en"], gpu=False, min_confidence=0.4):
    reader = easyocr.Reader(languages, gpu=gpu)

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

    results = [r for r in results if r[2] >= min_confidence]

    processed_results = []
    for bbox, text, prob in results:
        points = np.array(bbox, dtype=np.float32)

        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])

        shrink_factor = 0.95
        for i in range(len(points)):
            points[i, 0] = (points[i, 0] - center_x) * shrink_factor + center_x
            points[i, 1] = (points[i, 1] - center_y) * shrink_factor + center_y

        processed_results.append((points.tolist(), text, prob))

    return processed_results


def draw_boxes_on_image(image_path, boxes, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    result_image = image.copy()

    for bbox, _, _ in boxes:
        points = np.array(bbox, dtype=np.int32)

        cv2.polylines(
            result_image, [points], isClosed=True, color=(0, 255, 0), thickness=2
        )

    if output_path is None:
        input_path = Path(image_path)
        output_path = str(
            input_path.parent / f"{input_path.stem}_boxes{input_path.suffix}"
        )

    cv2.imwrite(output_path, result_image)
    print(f"Image with boxes saved to {output_path}")

    return result_image


def create_text_mask(image_path, boxes, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    height, width = image.shape[:2]
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    for bbox, _, _ in boxes:
        points = np.array(bbox, dtype=np.int32)

        cv2.fillPoly(mask, [points], color=(255, 255, 255))

    if output_path is None:
        input_path = Path(image_path)
        output_path = str(
            input_path.parent / f"{input_path.stem}_mask{input_path.suffix}"
        )

    cv2.imwrite(output_path, mask)
    print(f"Mask saved to {output_path}")

    return mask


def process_video(video_path, languages=["en"], gpu=False, min_confidence=0.4):
    video_file = Path(video_path)
    output_dir = video_file.parent / f"outputs/{video_file.stem}_output"

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

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    process_interval = max(1, int(fps / 2))
    print(
        f"Processing every {process_interval} frames, starting from frame {process_interval}"
    )

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    last_mask = None

    first_process_frame = process_interval

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = str(frames_dir / f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)

        should_process = (frame_count >= first_process_frame) and (
            (frame_count - first_process_frame) % process_interval == 0
        )

        if should_process:
            print(f"Processing frame {frame_count}")

            boxes = detect_text_boxes(
                frame_path,
                languages=languages,
                gpu=gpu,
                min_confidence=min_confidence,
            )

            boxes_path = str(boxes_dir / f"frame_{frame_count:04d}_boxes.jpg")
            draw_boxes_on_image(frame_path, boxes, output_path=boxes_path)

            mask_path = str(masks_dir / f"frame_{frame_count:04d}_mask.jpg")
            mask = create_text_mask(frame_path, boxes, output_path=mask_path)
            last_mask = mask
        else:
            if last_mask is not None:
                mask_path = str(final_masks_dir / f"frame_{frame_count:04d}_mask.jpg")
                cv2.imwrite(mask_path, last_mask)

        final_mask_path = str(final_masks_dir / f"frame_{frame_count:04d}_mask.jpg")
        if last_mask is not None:
            cv2.imwrite(final_mask_path, last_mask)
        else:
            empty_mask = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.imwrite(final_mask_path, empty_mask)

        frame_count += 1

    cap.release()

    final_mask_files = list(final_masks_dir.glob("*.jpg"))
    print(f"Generated {len(final_mask_files)} mask frames")

    if len(final_mask_files) != total_frames:
        print(
            f"WARNING: Number of mask frames ({len(final_mask_files)}) doesn't match original video ({total_frames})"
        )

    output_video = str(output_dir / f"{video_file.stem}_mask.mp4")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
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
    video_paths = [
        "v1.mp4",
        "v2.mp4",
        "v3.mp4",
        "v4.mp4",
    ]
    for video_path in video_paths:
        process_video(
            video_path,
            languages=["en"],
            gpu=False,
            min_confidence=0.4,
        )


if __name__ == "__main__":
    main()
