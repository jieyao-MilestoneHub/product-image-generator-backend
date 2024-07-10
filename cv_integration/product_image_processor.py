import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import torch
import logging

import sys
sys.path.append("./../")
from configs import model_type, checkpoint_path

# Configure logging to use the same configuration as the main application
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler('app.log', mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

class ProductImageProcessor:
    """
    A class to process product images, including segmentation and masking.
    """

    def __init__(self, model_type=model_type, checkpoint_path=checkpoint_path):
        """
        Initialize the ProductImageProcessor with the specified model type and checkpoint path.

        Args:
            model_type (str): Type of the model to use for segmentation.
            checkpoint_path (str): Path to the model checkpoint.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.predictor = SamPredictor(self.model)
        logger.info(f"Initialized ProductImageProcessor with model {model_type} on {self.device}")

    def segment_product(self, image_path):
        """
        Segment the product from the image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            tuple: A tuple containing the mask and the RGB image.
        """
        try:
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            points, labels = self.auto_select_points(image_rgb)
            self.predictor.set_image(image_rgb)
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True
            )
            best_mask_index = np.argmax(scores)
            mask = masks[best_mask_index].astype(np.uint8)
            logger.info(f"Segmentation completed for image {image_path}")
            return mask, image_rgb
        except Exception as e:
            logger.error(f"Error during segmentation for image {image_path}: {str(e)}")
            raise

    def auto_select_points(self, image):
        """
        Automatically select non-white area points in the image.

        Args:
            image (np.ndarray): The input RGB image.

        Returns:
            tuple: A tuple containing the selected points and their labels.
        """
        non_white = np.all(image < [245, 245, 245], axis=2)
        y_coords, x_coords = np.where(non_white)
        sample_indices = np.linspace(0, len(x_coords) - 1, num=5, dtype=int)
        points = np.column_stack((x_coords[sample_indices], y_coords[sample_indices]))
        labels = np.ones(len(points), dtype=int)
        logger.info("Auto-selected points for segmentation")
        return points, labels

    def apply_mask(self, image, mask):
        """
        Apply the mask to the image to make the background transparent.

        Args:
            image (np.ndarray): The input RGB image.
            mask (np.ndarray): The segmentation mask.

        Returns:
            Image: The resulting image with transparency.
        """
        try:
            bgra_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
            bgra_image[:, :, 3] = mask * 255
            logger.info("Applied mask to the image")
            return Image.fromarray(bgra_image)
        except Exception as e:
            logger.error(f"Error applying mask to image: {str(e)}")
            raise

    def save_image(self, image, path):
        """
        Save the image to the specified path.

        Args:
            image (Image): The image to save.
            path (str): The path to save the image.
        """
        try:
            image.save(path)
            logger.info(f"Saved image to {path}")
        except Exception as e:
            logger.error(f"Error saving image to {path}: {str(e)}")
            raise

if __name__ == "__main__":
    # Define paths for the input and output images
    image_path = "product_test/product_example_02.png"
    output_path = "product_test/product_example_02_trans.png"

    # Initialize the processor and perform segmentation and masking
    processor = ProductImageProcessor()
    mask, image_rgb = processor.segment_product(image_path)
    masked_image = processor.apply_mask(image_rgb, mask)
    processor.save_image(masked_image, output_path)
    print(f"Masked image saved to {output_path}")
