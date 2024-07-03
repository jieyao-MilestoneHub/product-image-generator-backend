import cv2
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import torch

class ProductImageProcessor:
    def __init__(self, model_type="vit_h", checkpoint_path="sam_vit_h_4b8939.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.predictor = SamPredictor(self.model)

    def segment_product(self, image_path):
        # 读取图像
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 自动选择非白色区域点
        points, labels = self.auto_select_points(image_rgb)

        # 设置图像并预测遮罩
        self.predictor.set_image(image_rgb)
        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )

        # 根据分数选择最佳遮罩
        best_mask_index = np.argmax(scores)
        mask = masks[best_mask_index].astype(np.uint8)

        return mask, image_rgb

    def auto_select_points(self, image):
        # 找到所有非白色像素点
        non_white = np.all(image < [245, 245, 245], axis=2)
        y_coords, x_coords = np.where(non_white)

        # 选择一些代表性点（例如：四分位点）
        sample_indices = np.linspace(0, len(x_coords) - 1, num=5, dtype=int)
        points = np.column_stack((x_coords[sample_indices], y_coords[sample_indices]))
        labels = np.ones(len(points), dtype=int)  # 所有选择点都视为前景点

        return points, labels

    def apply_mask(self, image, mask):
        # 应用遮罩到图像
        bgra_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
        bgra_image[:, :, 3] = mask * 255  # 转换遮罩为透明度通道
        return Image.fromarray(bgra_image)

    def save_image(self, image, path):
        image.save(path)
