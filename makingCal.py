import os
import cv2


def generate_rknn_calibration_dataset(
    video_path,
    output_dir="rknn_calib_images",
    num_frames=300,
    txt_filename="dataset.txt",
):
  # Create the output directory if it doesn't exist
  os.makedirs(output_dir, exist_ok=True)

  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    return

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  if total_frames <= 0:
    print("Error: Video appears to have no frames or an invalid frame count.")
    cap.release()
    return

  step = max(1, total_frames // num_frames)
  saved_count = 0

  txt_path = os.path.join(output_dir, txt_filename)

  print(f"Processing video: {video_path}")
  print(f"Targeting {num_frames} frames out of {total_frames} total frames...")

  with open(txt_path, "w") as f:
    for i in range(0, total_frames, step):
      if saved_count >= num_frames:
        break

      cap.set(cv2.CAP_PROP_POS_FRAMES, i)
      ret, frame = cap.read()

      if not ret:
        continue

      img_name = f"calib_frame_{saved_count:04d}.jpg"
      img_path = os.path.join(output_dir, img_name)
      cv2.imwrite(img_path, frame)

      f.write(os.path.abspath(img_path) + "\n")
      saved_count += 1

  cap.release()
  print(f"Done! Successfully saved {saved_count} frames.")
  print(f"Calibration text file generated at: {os.path.abspath(txt_path)}")


if __name__ == "__main__":
  # Change 'your_input_video.mp4' to your actual video filename
  TARGET_VIDEO = "flight2.mp4"

  generate_rknn_calibration_dataset(
      video_path=TARGET_VIDEO, num_frames=300, output_dir="rknn_calib_images"
  )
