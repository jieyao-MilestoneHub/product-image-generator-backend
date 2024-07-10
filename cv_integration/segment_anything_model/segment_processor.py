import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything import SamPredictor, sam_model_registry

import sys
sys.path.append("./../../")
from configs import model_type, checkpoint_path

class SegmentAnythingProcessor:
    def __init__(self, model_type=model_type, checkpoint_path=checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def load_image(self, path, convert_to="RGB"):
        return Image.open(path).convert(convert_to)

    def set_image(self, image_path):
        self.image = self.load_image(image_path, "RGB")
        self.image_np = np.array(self.image)
        self.predictor.set_image(self.image_np)
        self.height, self.width = self.image.size

    def segment_center_object(self):
        center_x, center_y = self.width // 2, self.height // 2
        input_point = np.array([[center_x, center_y]])
        input_label = np.array([1])

        masks, scores, _ = self.predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
        best_mask = masks[np.argmax(scores)]
        self.best_mask = best_mask
        return best_mask

    def save_masks(self, original_bg_path, transparent_bg_path):
        # 创建白色遮罩的原背景图
        mask_image_original = self.image.copy()
        mask_image_original_np = np.array(mask_image_original)
        mask = (self.best_mask * 255).astype(np.uint8)
        mask_3channel = np.stack([mask, mask, mask], axis=-1)  # 转换为3通道

        # 将白色遮罩应用到原图像
        mask_image_original_np = np.where(mask_3channel == 0, mask_image_original_np, [255, 255, 255])
        mask_image_original = Image.fromarray(mask_image_original_np.astype(np.uint8))
        mask_image_original.save(original_bg_path)

        # 创建白色遮罩的透明背景图
        mask_image_transparent = Image.new("RGBA", self.image.size)
        mask_white = Image.new("RGBA", self.image.size, (255, 255, 255, 255))
        mask_image_transparent = Image.composite(mask_white, mask_image_transparent, Image.fromarray(mask))
        mask_image_transparent.save(transparent_bg_path)


if __name__ == "__main__":

    processor = SegmentAnythingProcessor()

    # 设置图像
    processor.set_image(r"C:\Users\USER\Desktop\Develop\product-image-generator-backend\cv_integration\product_test\generated_image_20240709230238.png")

    # 分割中心物体
    processor.segment_center_object()

    # 保存结果遮罩图像
    processor.save_masks("original_bg_mask.png", "transparent_bg_mask.png")