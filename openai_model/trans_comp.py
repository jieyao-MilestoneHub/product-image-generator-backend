import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageChops

import random

class ImageProcessor:
    def __init__(self, foreground_path, background_path):
        self.foreground = Image.open(foreground_path).convert("RGBA")
        self.background = Image.open(background_path).convert("RGBA")

    def detect_container_areas(self, bg_image, fg_image):
        # 舉例: 分辨是否為容器
        gray = cv2.cvtColor(bg_image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # 使用形態學來操作清理圖像
        kernel = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

        # 可以使用標籤或連通組件來找到獨立的容器區域
        num_labels, labels_im = cv2.connectedComponents(cleaned)

        return labels_im

    def _find_best_position(self, fg, bg):
        fg_w, fg_h = fg.size
        bg_w, bg_h = bg.size
        bg_gray = bg.convert("L")
        bg_gray_np = np.array(bg_gray)
        
        # 獲得容器區
        container_areas = self.detect_container_areas(np.array(bg), np.array(fg))

        local_std = cv2.blur(np.square(bg_gray_np), (fg_w, fg_h))
        local_mean = cv2.blur(bg_gray_np, (fg_w, fg_h))
        mean_sq = np.square(local_mean)
        local_std = np.sqrt(local_std - mean_sq)

        # 評估每個區域標準差集容器可能性
        best_score = float('inf')
        best_position = (0, 0)
        for y in range(bg_h - fg_h + 1):
            for x in range(bg_w - fg_w + 1):
                # 定義分數：標準差低且在容器區域內得分高
                std_score = local_std[y:y+fg_h, x:x+fg_w].mean()
                container_score = container_areas[y:y+fg_h, x:x+fg_w].mean()  # 假設容器區域在 labels_im 中標記較高
                score = std_score - container_score  # 標準差低，容器得分高為佳

                if score < best_score:
                    best_score = score
                    best_position = (x, y)

        return best_position
    
    def _create_mask(self, fg):
        return fg.split()[3]
    
    def _blend_edges(self, image, mask):
        mask_resized = mask.resize(image.size, Image.Resampling.LANCZOS)
        blurred_mask = mask_resized.filter(ImageFilter.GaussianBlur(radius=5))
        composite_image = Image.composite(image, Image.new("RGBA", image.size), blurred_mask)
        return composite_image

    # 基本陰影
    def add_basic_shadow(self):
        shadow = Image.new("RGBA", self.background.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(shadow)
        position = self._find_best_position(self.foreground, self.background)
        fg_w, fg_h = self.foreground.size
        offset = 5  # 陰影偏移量
        shadow_box = [position[0] + offset, position[1] + fg_h - 10, position[0] + fg_w + offset, position[1] + fg_h]
        draw.ellipse(shadow_box, fill=(0, 0, 0, 128))  # 半透明黑色陰影
        self.background = Image.alpha_composite(self.background, shadow)

    # 柔化邊緣陰影
    def add_soft_shadow(self):
        shadow = Image.new("RGBA", self.background.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(shadow)
        position = self._find_best_position(self.foreground, self.background)
        fg_w, fg_h = self.foreground.size
        offset = 5
        shadow_box = [position[0] + offset, position[1] + fg_h - 5, position[0] + fg_w + offset, position[1] + fg_h]
        draw.ellipse(shadow_box, fill=(0, 0, 0, 128))
        blurred_shadow = shadow.filter(ImageFilter.GaussianBlur(10))
        self.background = Image.alpha_composite(self.background, blurred_shadow)

    # 光暈效果
    def add_glow_effect(self):
        glow = Image.new("RGBA", self.background.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(glow)
        position = self._find_best_position(self.foreground, self.background)
        fg_w, fg_h = self.foreground.size
        glow_box = [position[0] - 10, position[1] - 10, position[0] + fg_w + 10, position[1] + fg_h + 10]
        draw.ellipse(glow_box, fill=(255, 255, 255, 30))
        blurred_glow = glow.filter(ImageFilter.GaussianBlur(15))
        self.background = Image.alpha_composite(self.background, blurred_glow)

    def composite_images(self, scale=1.0):
        # 動態調整產品圖片大小
        original_size = self.foreground.size
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        resized_foreground = self.foreground.resize(new_size, Image.Resampling.LANCZOS)

        # 最佳位置
        position = self._find_best_position(resized_foreground, self.background)
        
        # 提取產品圖 Alpha 通道作為遮罩
        mask = resized_foreground.split()[3]

        # 基於產品圖調整遮罩大小
        mask_resized = mask.resize(new_size, Image.Resampling.LANCZOS)

        # 判斷產品置入位置是否在底部
        if position[1] > self.background.height - self.background.height * 0.1:
            # 若在底部，不增加效果
            print("Position is at the bottom, no effects will be added.")
        else:
            # 不在底部，隨機選擇一個效果
            effects = [self.add_basic_shadow, self.add_soft_shadow, self.add_glow_effect]
            chosen_effect = random.choice(effects)
            chosen_effect()  # 應用選擇的效果
            print(f"Applied effect: {chosen_effect.__name__}")

        # 黏貼產品圖至背景圖，使用調整的遮罩，確保產品圖在所有效果頂部
        self.background.paste(resized_foreground, position, mask_resized)

    def enhance_foreground_saturation(self):
        enhancer = ImageEnhance.Color(self.foreground)
        self.foreground = enhancer.enhance(2)

    def save_image(self, path):
        self.background.save(path)
