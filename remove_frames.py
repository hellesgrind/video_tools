import subprocess
import os


def remove_first_frames(input_file, output_file, n_frames):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return False

    fps_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_file,
    ]

    try:
        fps_output = subprocess.check_output(fps_cmd, universal_newlines=True).strip()
        if "/" in fps_output:
            num, den = map(int, fps_output.split("/"))
            fps = num / den
        else:
            fps = float(fps_output)

        start_time = n_frames / fps

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-i",
            input_file,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            output_file,
        ]

        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Successfully removed first {n_frames} frames from video.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error during processing: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


if __name__ == "__main__":
    files = ["1.mp4", "2.mp4", "3.mp4", "4.mp4", "5.mp4"]
    for file in files:
        input_file = file
        output_file = f"{file.split('.')[0]}_output.mp4"
        n_frames = 5
        remove_first_frames(input_file, output_file, n_frames)
