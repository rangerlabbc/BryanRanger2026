# imports
import os
import cv2
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean


# =============================================================================
# Helper functions 
# =============================================================================

def is_abd(file_path):
  """
  Check if a file is an abdomen ultrasound image or video.
  Return True if file belongs to an Abd folder, False otherwise.
  """
  folder_segments = file_path.parent.name.split('_')
  return "Abd" in folder_segments


# =============================================================================
# Non-abdomen cropping
# =============================================================================

def process_frame(frame):
  """
  Detects and returns crop coordinates for the non-black content in a frame.
  """

  # get frame dimensions
  h_orig, w_orig = frame.shape[:2]

  # convert to grayscale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # create a binary mask where non-black pixels are white
  _, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY) #threshold set to 10 for stricter black removal

  # ignore graphics on the sides by only using the middle 81.8% of the mask
  side_margin = int(w_orig * 0.091)
  binary[:, :side_margin] = 0 #erase left 9.1%
  binary[:, w_orig - side_margin:] = 0 #erase right 9.1%

  # find coordinates of non-black pixels
  coords = cv2.findNonZero(binary) #returns x, y coordinates of all non-black pixels
  if coords is None:
    return None

  # get the bounding box of the non-black area
  x, y, w, h = cv2.boundingRect(coords)

  x_start = max(0, x)
  y_start = max(0, y)
  x_end = min(w_orig, x + w)
  y_end = min(h_orig, y + h)

  # return cropping coords
  return x_start, y_start, x_end, y_end


def process_non_abd(source_base, target_base, cutoff_date=None):
  """
  Crops all non-abdomen images and videos in source_base and save them to a matching folder structure in target_base.
  Skips hidden files, abdomen files, and files older than the cutoff date. 
  """

  source_path = Path(source_base)
  target_path = Path(target_base)

  if cutoff_date is None:
    cutoff_date = datetime(2026, 6, 20)

  # extensions we want to process
  IMAGE_EXTS = ('.jpeg', '.jpg')
  VIDEO_EXTS = ('.mp4',)

  for root, dirs, files in os.walk(source_path):
    for file in files:
      file_path = Path(root) / file
      ext = file_path.suffix.lower()

      # skip hidden files
      if file.startswith('.'):
        continue

      # skip abd files
      if is_abd(file_path):
        continue

      # skip old files
      if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
        continue

      # create matching folder structure
      relative_path = file_path.relative_to(source_path)
      output_path = target_path / relative_path
      output_path.parent.mkdir(parents=True, exist_ok=True)
    
      # check if the cropped file already exists
      if output_path.exists():
          print(f"File already processed: {relative_path}")
          continue
      
      # process media
      try:
        # process images
        if ext in IMAGE_EXTS:
          image = cv2.imread(str(file_path))
          if image is None:
            print(f"Error: Unable to load image {file_path}")
            continue

          crop_coords = process_frame(image)
          if crop_coords is None:
            print(f"Error: Unable to crop image, skipping: {relative_path}")

          x_start, y_start, x_end, y_end = crop_coords
          cropped_image = image[y_start:y_end, x_start:x_end]

          success = cv2.imwrite(str(output_path), cropped_image)
          if success:
            print(f"Cropped image: {relative_path}")
          else:
            print(f"Failed to save image: {relative_path}")

        # process videos
        elif ext in VIDEO_EXTS:
          cap = cv2.VideoCapture(str(file_path))
          if not cap.isOpened():
            print(f"Error: Unable to open video {file_path}")
            continue

          fps = cap.get(cv2.CAP_PROP_FPS)
          ret, first_frame = cap.read()
          if not ret or first_frame is None:
            print(f"Error: Unable to read first frame from video {file_path}")
            cap.release()
            continue
        
          # compute crop coordinates from the first frame, apply same coordinates to all frames
          crop_coords = process_frame(first_frame)
          if crop_coords is None:
            print(f"Error: Unable to crop video, skipping: {relative_path}")
            cap.release()
            continue

          x_start, y_start, x_end, y_end = crop_coords
          fourcc = cv2.VideoWriter_fourcc(*'mp4v')
          out = cv2.VideoWriter(str(output_path), fourcc, fps, (x_end - x_start, y_end - y_start))

          # write first frame and apply same coordinates to all remaining frames
          out.write(first_frame[y_start:y_end, x_start:x_end])
          while True:
            ret, frame = cap.read()
            if not ret:
              break # end of video
            out.write(frame[y_start:y_end, x_start:x_end])

          cap.release()
          out.release()
          print(f"Cropped video: {relative_path}")

      except Exception as e:
        print(f"Skipped {file} due to error: {e}")


# =============================================================================
# Abdomen cropping
# =============================================================================

def is_blue_tinted(image):
  """
  Returns True if the image has color saturation (Ref1 type),
  False if grayscale/no saturation (Ref2 and MSK type).
  """
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  fan_hsv = hsv[gray > 10]
  if len(fan_hsv) == 0:
    return False
  sat_mean = fan_hsv[:, 1].mean()
  return (sat_mean > 20) # tune if needed


def is_MSK(file_path):
  """
  Returns True if the file is from an MSK folder (Ref3 type). Returns False otherwise (Ref2 type).
  Add folder names to MSK_FOLDERS list as needed. 
  """
  MSK_FOLDERS = [
  'NP015_Abd',
  'P001_Abd_T1',
  'P005_Abd_T1',
  'P006_Abd_T1',
  'P054_Abd_T2',
  'P055_Abd_T2',
  'P059_Abd_T2',
  'P060_Abd_T2',
  'P062_Abd_T2',
  # add more as needed
  ]
  return file_path.parent.name in MSK_FOLDERS


def get_logo_coords(image):
  """
  Finds and returns the bounding rect coords of the Clarius logo.
  Uses HSV bounds for logo detection.

  Tune the HSV bounds if needed. 
  hue: 0-15 (red-orange), saturation: 10-255, value: 10-255
  """
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  lower = np.array([0, 10, 10], dtype = np.uint8)
  upper = np.array([15, 255, 255], dtype = np.uint8)
  logo_mask = cv2.inRange(hsv, lower, upper).astype(bool)

  coords = cv2.findNonZero(logo_mask.astype(np.uint8))
  if coords is None:
    return None
  return cv2.boundingRect(coords)


def compute_fan_mask(reference_path):
  """
  Computes a filled fan mask and bounding rect from reference image.
  Returns a boolean mask (fan_mask_bool, x, y, w, h).
  """

  ref = cv2.imread(str(reference_path))
  if ref is None:
    raise ValueError(f"Could not load reference image: {reference_path}")

  h, w = ref.shape[:2]
  ref_clean = ref.copy()

  # remove logo
  logo_coords = get_logo_coords(ref_clean)
  if logo_coords is not None:
    lx, ly, lw, lh = logo_coords
    ref_clean[ly:ly+lh, lx:lx+lw] = 0

  # zero out ruler on the right (scan columns from right until bright pixels found)
  gray_right = cv2.cvtColor(ref_clean, cv2.COLOR_BGR2GRAY)
  ruler_col = w  # default: no ruler found
  for col in range(w - 1, int(w * 0.7), -1):
    if (gray_right[:, col] > 200).sum() > 5:
      ruler_col = col
      break
  ref_clean[:, max(0, ruler_col - 20):] = 0

  # threshold to find fan region
  gray_clean = cv2.cvtColor(ref_clean, cv2.COLOR_BGR2GRAY)
  _, binary = cv2.threshold(gray_clean, 5, 255, cv2.THRESH_BINARY)

  # find largest contour — this is the fan
  contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if not contours:
    raise ValueError(f"Could not detect fan contour in reference image: {reference_path}")

  largest = max(contours, key=cv2.contourArea)
  x, y, bw, bh = cv2.boundingRect(largest)

  # build filled fan mask (True inside the fan contour)
  fan_mask = np.zeros((h, w), dtype=np.uint8)
  cv2.drawContours(fan_mask, [largest], -1, 255, thickness=cv2.FILLED)

  print(f"Fan mask computed from {reference_path}: bounding rect x={x}, y={y}, w={bw}, h={bh}")
  return fan_mask.astype(bool), x, y, bw, bh


def process_abd_frame(frame, fan_mask, x, y, bw, bh, logo_coords):
  """
  Applies the fan mask to a single frame and crops to the bounding rect (x, y, bw, bh).
  Returns cropped frame as numpy array.
  """
  result = frame.copy()

  # remove logo
  if logo_coords is not None:
    lx, ly, lw, lh = logo_coords
    result[ly:ly+lh, lx:lx+lw] = 0

  # crop to bounding rect
  return result[y:y+bh, x:x+bw]


def process_abd(source_base, target_base, reference_path1, reference_path2, cutoff_date=None):
  """
  Crops all abdomen images and videos in source_base and save them to a matching folder structure in target_base.
  Skips hidden files, abdomen files, and files older than the cutoff date. 
  """
  source_path = Path(source_base)
  target_path = Path(target_base)

  if cutoff_date is None:
    cutoff_date = datetime(2026, 7, 9)

  # extensions we want to process
  IMAGE_EXTS = ('.jpeg', '.jpg')
  VIDEO_EXTS = ('.mp4',)

  # compute boths global masks once
  fan_mask1, fx1, fy1, fbw1, fbh1 = compute_fan_mask(reference_path1)
  fan_mask2, fx2, fy2, fbw2, fbh2 = compute_fan_mask(reference_path2)
  ref1 = cv2.imread(str(reference_path1))
  ref2 = cv2.imread(str(reference_path2))
  ref1_h, ref1_w = ref1.shape[:2]
  ref2_h, ref2_w = ref2.shape[:2]

  # initialize MSK references
  reference_path3 = None
  fan_mask3 = fx3 = fy3 = fbw3 = fbh3 = None

  for root, dirs, files in os.walk(source_path):
    for file in files:
      file_path = Path(root) / file
      ext = file_path.suffix.lower()

      # skip hidden files
      if file.startswith('.'):
        continue

      # skip non abd files
      if not is_abd(file_path):
        continue

      # skip old files
      if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
        continue

      # create matching folder structure
      relative_path = file_path.relative_to(source_path)
      output_path = target_path / relative_path
      output_path.parent.mkdir(parents=True, exist_ok=True)

      # check if the cropped file already exists
      if output_path.exists():
          print(f"File already processed: {relative_path}")
          continue

      # process media
      try:
        # process abd images
        if ext in IMAGE_EXTS:
          image = cv2.imread(str(file_path))
          if image is None:
            print(f"Error: Unable to load image {file_path}")
            continue

          # get logo coords once
          logo_coords = get_logo_coords(image)

          # route to the correct mask based on tint or MSK scan
          if is_blue_tinted(image):
            fan_mask, fx, fy, fbw, fbh = fan_mask1, fx1, fy1, fbw1, fbh1
          elif is_MSK(file_path):
            # use the first image in this MSK folder as the reference for the fan mask
            if reference_path3 is not None and file_path.parent.name == reference_path3.parent.name:
              fan_mask, fx, fy, fbw, fbh = fan_mask3, fx3, fy3, fbw3, fbh3
            else:
              reference_path3 = file_path
              fan_mask3, fx3, fy3, fbw3, fbh3 = compute_fan_mask(reference_path3)
              fan_mask, fx, fy, fbw, fbh = fan_mask3, fx3, fy3, fbw3, fbh3
          else:
            fan_mask, fx, fy, fbw, fbh = fan_mask2, fx2, fy2, fbw2, fbh2

          processed = process_abd_frame(image, fan_mask, fx, fy, fbw, fbh, logo_coords)
          success = cv2.imwrite(str(output_path), processed)
          if success:
            print(f"Processed Abd image: {relative_path}")
          else:
            print(f"Failed to process Abd image: {relative_path}")

        # process abd videos
        elif ext in VIDEO_EXTS:
          cap = cv2.VideoCapture(str(file_path))
          if not cap.isOpened():
            print(f"Error: Unable to open video {file_path}")
            continue

          fps = cap.get(cv2.CAP_PROP_FPS)
          w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
          h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

          ret, first_frame = cap.read()
          if not ret or first_frame is None:
            print(f"Error: Unable to read first frame from video {file_path}")
            cap.release()
            continue
          
          # get video logo coords once
          logo_coords = get_logo_coords(first_frame)

          # route to the correct mask based on tint or MSK scan
          if is_blue_tinted(first_frame):
            fan_mask, fx, fy, fbw, fbh = fan_mask1, fx1, fy1, fbw1, fbh1
          elif is_MSK(file_path):
            if reference_path3 is not None and file_path.parent.name == reference_path3.parent.name:
              fan_mask, fx, fy, fbw, fbh = fan_mask3, fx3, fy3, fbw3, fbh3
            else:
              reference_path3 = file_path
              fan_mask3, fx3, fy3, fbw3, fbh3 = compute_fan_mask(reference_path3)
              fan_mask, fx, fy, fbw, fbh = fan_mask3, fx3, fy3, fbw3, fbh3
          else:
            fan_mask, fx, fy, fbw, fbh = fan_mask2, fx2, fy2, fbw2, fbh2

          # write to a temp file first then rename to avoid corrupt output on failure
          temp_path = output_path.with_suffix('.temp.mp4')
          fourcc = cv2.VideoWriter_fourcc(*'mp4v')
          out = cv2.VideoWriter(str(temp_path), fourcc, fps, (fbw, fbh))

          # write first frame and apply same coordinates to all remaining frames
          out.write(process_abd_frame(first_frame, fan_mask, fx, fy, fbw, fbh, logo_coords))

          # process remaining frames
          while True:
            ret, frame = cap.read()
            if not ret:
              break # end of video
            out.write(process_abd_frame(frame, fan_mask, fx, fy, fbw, fbh, logo_coords))

          cap.release()
          out.release()
          temp_path.replace(output_path)
          print(f"Processed Abd video: {relative_path}")

      except Exception as e:
        print(f"Skipped {file_path} due to error: {e}")


# =============================================================================
# Bad contact cropping 
# Removes bad-contact artifact regions on the left/right edges of non-abd images or video frames. 
# Operates on already cropped files, intended to further crop selected frames from videos. 
# =============================================================================

CONTENT_THRESHOLD = 175  # tune if needed — real content has 400+, bad contact has <100

def crop_frame_contacts(frame, threshold=CONTENT_THRESHOLD):
  """
  Detect bad-contact regions on left and right edges and crop them.
  Returns (cropped_frame, was_cropped, left_crop, right_crop).
  """
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  h, w = gray.shape

  # scan from right
  right_crop = w
  for col in range(w - 1, 20, -1):
    if (gray[:, col] > 10).sum() >= threshold:
      right_crop = col + 1
      break

  # scan from left
  left_crop = 0
  for col in range(0, w - 20):
    if (gray[:, col] > 10).sum() >= threshold:
      left_crop = col
      break

  was_cropped = (left_crop > 0) or (right_crop < w)
  cropped = frame[:, left_crop:right_crop]
  return cropped, was_cropped, left_crop, right_crop


def crop_contact_images(source_base, threshold=CONTENT_THRESHOLD):
  """
  Scans all non-abdomen images in source_base, detects and removes bad-contact regions. 
  Overwrites the original files. 
  """
  source_path = Path(source_base)
  IMAGE_EXTS = ('.jpeg', '.jpg')

  for root, dirs, files in os.walk(source_path):
    for file in files:
      file_path = Path(root) / file
      ext = file_path.suffix.lower()

      # skip hidden files
      if file.startswith('.'):
        continue

      # skip non-images
      if ext not in IMAGE_EXTS:
        continue

      #skip abd files
      if is_abd(file_path):
        continue

      try:
        image = cv2.imread(str(file_path))
        if image is None:
          print(f"Error: Unable to load image {file_path}")
          continue

        cropped, was_cropped, left_crop, right_crop = crop_frame_contacts(image, threshold)
        if not was_cropped: 
          print(f"No cropping needed: {file_path.relative_to(source_path)}")
          continue
        success = cv2.imwrite(str(file_path), cropped)
        if success:
          print(f"Cropped: {file_path.relative_to(source_path)} - left: {left_crop}px, right: {(image.shape[1] - right_crop)}px")
        else:
          print(f"Failed to save: {file_path.relative_to(source_path)}")

      except Exception as e:
        print(f"Skipped {file} due to error: {e}")
