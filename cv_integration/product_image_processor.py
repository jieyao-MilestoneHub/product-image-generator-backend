import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import torch

import sys
sys.path.append("./../")
from configs import model_type, checkpoint_path

class ProductImageProcessor:
    def __init__(self, model_type=model_type, checkpoint_path=checkpoint_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.predictor = SamPredictor(self.model)

    # 分割產品圖像
    def segment_product(self, image_path):
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
        return mask, image_rgb

    # 自動選擇非白色區域點
    def auto_select_points(self, image):
        non_white = np.all(image < [245, 245, 245], axis=2)
        y_coords, x_coords = np.where(non_white)
        sample_indices = np.linspace(0, len(x_coords) - 1, num=5, dtype=int)
        points = np.column_stack((x_coords[sample_indices], y_coords[sample_indices]))
        labels = np.ones(len(points), dtype=int)
        return points, labels

    # 應用遮罩
    def apply_mask(self, image, mask):
        bgra_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
        bgra_image[:, :, 3] = mask * 255
        return Image.fromarray(bgra_image)

    # 保存圖片
    def save_image(self, image, path):
        image.save(path)

if __name__ == "__main__":
    image_path = "product_test/product_example_02.png"
    output_path = "product_test/product_example_02_trans.png"

    processor = ProductImageProcessor()
    mask, image_rgb = processor.segment_product(image_path)
    masked_image = processor.apply_mask(image_rgb, mask)
    processor.save_image(masked_image, output_path)
    print(f"Masked image saved to {output_path}")