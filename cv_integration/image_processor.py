import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import random

class ImageProcessor:
    def __init__(self, foreground_path, background_path):
        self.foreground = Image.open(foreground_path).convert("RGBA")
        self.background = Image.open(background_path).convert("RGBA")

    # 檢測清晰且一致的區域
    def detect_clear_and_consistent_areas(self, bg_image, fg_image):
        gray = cv2.cvtColor(bg_image, cv2.COLOR_BGR2GRAY)
        fg_h, fg_w, _ = fg_image.shape
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = cv2.blur(np.square(laplacian), (fg_w, fg_h))
        sharpness_mask = laplacian_var > laplacian_var.mean()
        local_mean = cv2.blur(gray, (fg_w, fg_h))
        local_std = np.sqrt(cv2.blur(np.square(gray), (fg_w, fg_h)) - np.square(local_mean))
        consistency_mask = local_std < local_std.mean()
        combined_mask = np.logical_and(sharpness_mask, consistency_mask).astype(np.uint8) * 255
        return combined_mask

    # 找到前景圖像的最佳位置
    def _find_best_position(self, fg, bg):
        fg_w, fg_h = fg.size
        bg_w, bg_h = bg.size
        bg_np = np.array(bg)
        clear_consistent_areas = self.detect_clear_and_consistent_areas(bg_np, np.array(fg))
        best_score = float('inf')
        best_position = (0, 0)
        max_y = bg_h - fg_h
        min_y = max_y - bg_h // 6
        for y in range(min_y, max_y + 1):
            for x in range(bg_w - fg_w + 1):
                area = clear_consistent_areas[y:y+fg_h, x:x+fg_w]
                score = -area.mean()
                if score < best_score:
                    best_score = score
                    best_position = (x, y)
        return best_position

    # 添加基本陰影
    def add_basic_shadow(self):
        shadow = Image.new("RGBA", self.background.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(shadow)
        position = self._find_best_position(self.foreground, self.background)
        fg_w, fg_h = self.foreground.size
        offset = 5
        shadow_box = [position[0] + offset, position[1] + fg_h - 10, position[0] + fg_w + offset, position[1] + fg_h]
        draw.ellipse(shadow_box, fill=(0, 0, 0, 128))
        self.background = Image.alpha_composite(self.background, shadow)

    # 添加柔化邊緣陰影
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

    # 合成圖像
    def composite_images(self, scale=1.0):
        original_size = self.foreground.size
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        resized_foreground = self.foreground.resize(new_size, Image.Resampling.LANCZOS)
        position = self._find_best_position(resized_foreground, self.background)
        mask = resized_foreground.split()[3]
        mask_resized = mask.resize(new_size, Image.Resampling.LANCZOS)
        if (position[1] + resized_foreground.size[1]) > (self.background.height * 5 / 6):
            print("Position is at the bottom 1/6, no effects will be added.")
        else:
            effects = [self.add_basic_shadow, self.add_soft_shadow]
            chosen_effect = random.choice(effects)
            chosen_effect()
            print(f"Applied effect: {chosen_effect.__name__}")
        self.background.paste(resized_foreground, position, mask_resized)

    # 增強前景圖像的飽和度
    def enhance_foreground_saturation(self):
        enhancer = ImageEnhance.Color(self.foreground)
        self.foreground = enhancer.enhance(7)

    # 保存圖像
    def save_image(self, path):
        self.background.save(path)
