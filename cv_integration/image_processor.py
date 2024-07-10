import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import random

class ImageProcessor:
    def __init__(self, foreground_path, background_path):
        self.foreground = Image.open(foreground_path).convert("RGBA")
        self.background = Image.open(background_path).convert("RGBA")

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

    def _find_best_position(self, fg, bg):
        fg_w, fg_h = fg.size
        bg_w, bg_h = bg.size
        # Calculate the horizontal center position
        center_x = (bg_w - fg_w) // 2
        # Adjust vertical position so the bottom of fg is at the bottom 1/6th of bg
        center_y = (8 * bg_h // 9) - fg_h
        return (center_x, center_y)

    # def add_basic_shadow(self):
    #     shadow = Image.new("RGBA", self.background.size, (0, 0, 0, 0))
    #     draw = ImageDraw.Draw(shadow)
    #     position = self._find_best_position(self.foreground, self.background)
    #     fg_w, fg_h = self.foreground.size
    #     offset = 5
    #     shadow_box = [position[0] + offset, position[1] + fg_h - 10, position[0] + fg_w + offset, position[1] + fg_h]
    #     draw.ellipse(shadow_box, fill=(0, 0, 0, 128))
    #     self.background = Image.alpha_composite(self.background, shadow)

    # def add_soft_shadow(self):
    #     shadow = Image.new("RGBA", self.background.size, (0, 0, 0, 0))
    #     draw = ImageDraw.Draw(shadow)
    #     position = self._find_best_position(self.foreground, self.background)
    #     fg_w, fg_h = self.foreground.size
    #     offset = 5
    #     shadow_box = [position[0] + offset, position[1] + fg_h - 5, position[0] + fg_w + offset, position[1] + fg_h]
    #     draw.ellipse(shadow_box, fill=(0, 0, 0, 128))
    #     blurred_shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    #     self.background = Image.alpha_composite(self.background, blurred_shadow)

    def composite_images(self, scale=1.0):
        original_size = self.foreground.size
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        resized_foreground = self.foreground.resize(new_size, Image.Resampling.LANCZOS)
        
        # Ensure resized foreground does not exceed background boundaries
        if resized_foreground.width > self.background.width or resized_foreground.height > self.background.height:
            scale = min(self.background.width / original_size[0], self.background.height / original_size[1])
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            resized_foreground = self.foreground.resize(new_size, Image.Resampling.LANCZOS)
        
        position = self._find_best_position(resized_foreground, self.background)
        mask = resized_foreground.split()[3]
        mask_resized = mask.resize(new_size, Image.Resampling.LANCZOS)
        # if (position[1] + resized_foreground.size[1]) > (self.background.height * 5 / 6):
        #     print("Position is at the bottom 1/6, no effects will be added.")
        # else:
        #     effects = [self.add_basic_shadow]
        #     chosen_effect = random.choice(effects)
        #     chosen_effect()
        #     print(f"Applied effect: {chosen_effect.__name__}")
        self.background.paste(resized_foreground, position, mask_resized)

    def enhance_foreground_saturation(self):
        enhancer = ImageEnhance.Color(self.foreground)
        self.foreground = enhancer.enhance(7)

    def save_image(self, path):
        self.background.save(path)

if __name__ == "__main__":
    foreground_path = r"C:\Users\USER\Desktop\Develop\product-image-generator-backend\cv_integration\product_test\product_example_01_trans.png"
    background_path = r"C:\Users\USER\Desktop\Develop\sd_sagemaker\content\generated_images\text-image\generated_image_20240710031944.png"
    output_path = "output.png"

    processor = ImageProcessor(foreground_path, background_path)
    processor.enhance_foreground_saturation()
    processor.composite_images(scale=2.5)
    processor.save_image(output_path)
    print(f"Composite image saved to {output_path}")
