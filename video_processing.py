import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import tempfile
import platform
from typing import Dict, Tuple

# Define colors for each segmentation category
CATEGORY_COLORS = {
    0: (0, 0, 0),       # background - black
    1: (255, 0, 0),     # hair - blue
    2: (0, 255, 0),     # body-skin - green
    3: (0, 0, 255),     # face-skin - red
    4: (255, 255, 0),   # clothes - yellow
    5: (255, 0, 255),   # others (accessories) - magenta
}

CATEGORY_NAMES = {
    0: "background",
    1: "hair",
    2: "body-skin",
    3: "face-skin",
    4: "clothes",
    5: "others"
}

class VideoProcessor:
    def __init__(self, model_path: str):
        """Initialize the video processor with the tflite model."""
        self.model_path = model_path
        self.segmenter = None
        self._initialize_segmenter()

    def _initialize_segmenter(self):
        """Initialize the MediaPipe ImageSegmenter."""
        if platform.system() == "Linux":
            base_options = python.BaseOptions(
                model_asset_path=self.model_path,
                delegate=python.BaseOptions.Delegate.GPU
            )
        else:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
        
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def apply_segmentation_blur(self, frame: np.ndarray, category_mask: np.ndarray) -> np.ndarray:
        """
        Apply a blur effect to the frame, keeping hair and face-skin sharp.

        Args:
            frame: Input frame (BGR format)
            category_mask: Segmentation mask from the model

        Returns:
            Processed frame with blur effect.
        """
        # Categories to keep sharp: 1 for hair, 3 for face-skin
        sharp_categories = [1, 3]

        # Create a mask for the sharp categories
        sharp_mask = np.isin(category_mask, sharp_categories)

        # Create a heavily blurred version of the frame
        blurred_frame = cv2.GaussianBlur(frame, (151, 151), 0)

        # Combine the sharp and blurred parts
        # We need to expand the mask's dimensions to match the frame's 3 channels
        sharp_mask_3d = np.stack([sharp_mask] * 3, axis=-1)
        output_frame = np.where(sharp_mask_3d, frame, blurred_frame)

        return output_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with segmentation and color application.

        Args:
            frame: Input frame in BGR format

        Returns:
            Processed frame with segmentation colors applied
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run segmentation
        segmentation_result = self.segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask

        # Apply blur based on segmentation
        processed_frame = self.apply_segmentation_blur(frame, category_mask.numpy_view())

        return processed_frame

    def process_video(self, input_path: str, output_path: str, progress_callback=None) -> bool:
        """
        Process a video file with selfie segmentation.

        Args:
            input_path: Path to input video file
            output_path: Path to output video file
            progress_callback: Optional callback function for progress updates

        Returns:
            True if successful, False otherwise
        """
        try:
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return False

            # Get video properties
            fps = 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                processed_frame = self.process_frame(frame)

                # Write processed frame
                out.write(processed_frame)

                frame_count += 1

                # Update progress
                if progress_callback and total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    progress_callback(progress)

            # Release resources
            cap.release()
            out.release()

            return True

        except Exception as e:
            print(f"Error processing video: {e}")
            return False

    def get_segmentation_info(self) -> Dict:
        """Get information about the segmentation categories."""
        return {
            "categories": CATEGORY_NAMES,
            "colors": CATEGORY_COLORS,
            "model_path": self.model_path
        }

# Global processor instance
processor = None

def get_processor() -> VideoProcessor:
    """Get or create the global video processor instance."""
    global processor
    if processor is None:
        model_path = os.path.join(os.path.dirname(__file__), "model", "selfie_multiclass_256x256.tflite")
        processor = VideoProcessor(model_path)
    return processor

def process_video_file(input_path: str, output_path: str = None, progress_callback=None) -> Tuple[bool, str]:
    """
    Process a video file and return the output path.

    Args:
        input_path: Path to input video file
        output_path: Path to output video file (optional)
        progress_callback: Optional progress callback function

    Returns:
        Tuple of (success, output_path)
    """
    if output_path is None:
        # Create temporary output file
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"processed_{os.path.basename(input_path)}")

    processor = get_processor()
    success = processor.process_video(input_path, output_path, progress_callback)

    return success, output_path
