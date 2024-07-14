import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T
import os
import sys

# Set system path
sys.path.append("./../")
from configs import deepfill_model_path, import_path
sys.path.append(import_path)

class DeepFill:
    def __init__(self, model_path, device=None):
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        generator_state_dict = torch.load(model_path, map_location=self.device)['G']
        if 'stage1.conv1.conv.weight' in generator_state_dict:
            from deepfill_v2_model.model.networks import Generator
        else:
            from deepfill_v2_model.model.networks_tf import Generator
        self.generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(self.device)
        self.generator.load_state_dict(generator_state_dict, strict=True)
        self.generator.eval()

    def inpaint(self, image, mask):
        image = T.ToTensor()(image).unsqueeze(0)
        mask = T.ToTensor()(mask).unsqueeze(0)
        image, mask = self._preprocess_image_and_mask(image, mask)
        image_masked = image * (1. - mask)
        x = torch.cat([image_masked, torch.ones_like(image_masked)[:, 0:1, :, :], mask], dim=1)
        with torch.inference_mode():
            _, x_stage2 = self.generator(x, mask)
        image_inpainted = image * (1. - mask) + x_stage2 * mask
        return self._postprocess_image(image_inpainted)

    def _preprocess_image_and_mask(self, image, mask):
        _, h, w = image.shape[1:]
        grid = 8
        image = image[:, :3, :h // grid * grid, :w // grid * grid]
        mask = mask[:, :1, :h // grid * grid, :w // grid * grid]
        image = (image * 2 - 1.).to(self.device)
        mask = (mask > 0.5).float().to(self.device)
        return image, mask

    def _postprocess_image(self, image_inpainted):
        img_out = ((image_inpainted[0].permute(1, 2, 0) + 1) * 127.5).to('cpu', torch.uint8)
        return Image.fromarray(img_out.numpy())

class ImageReplacer:
    def __init__(self, deepfill_model_path, save_intermediate=False):
        self.model = maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()
        self.deepfill = DeepFill(deepfill_model_path)
        self.save_intermediate = save_intermediate

    def process_image(self, base_image_path, overlay_image_path, output_path, target_size, scale_factor):
        base_image = Image.open(base_image_path).convert("RGBA")
        overlay_image = Image.open(overlay_image_path).convert("RGBA")

        scaled_base_image = base_image.resize(target_size, Image.Resampling.LANCZOS)
        self.replace_object(base_image, scaled_base_image, overlay_image, output_path, scale_factor)

    def replace_object(self, original_image, scaled_image, overlay_image, output_path, scale_factor):
        base_mask = self.refine_mask(self.get_mask(original_image))
        if base_mask is None:
            print("No object detected to replace.")
            return

        if self.save_intermediate:
            original_image.save("output_test/original_image.png")
            mask_image = Image.fromarray((base_mask * 255).astype(np.uint8)).convert("L")
            mask_image.save(f"output_test/mask_image_{os.path.basename(output_path)}.png")

        inpainted_image = self.deepfill.inpaint(original_image, Image.fromarray((base_mask * 255).astype(np.uint8)).convert("L"))
        if self.save_intermediate:
            inpainted_image.save(f"output_test/inpainted_image_{os.path.basename(output_path)}.png")

        inpainted_resized = inpainted_image.resize(scaled_image.size, Image.Resampling.LANCZOS)
        if self.save_intermediate:
            inpainted_resized.save(f"output_test/inpainted_resized_{os.path.basename(output_path)}.png")

        transformed_overlay_image, mask_bounds = self.adjust_overlay_size(overlay_image, base_mask, scale_factor, original_image.size, scaled_image.size)
        if self.save_intermediate:
            transformed_overlay_image.save(f"output_test/transformed_overlay_image_{os.path.basename(output_path)}.png")

        result_image = self.merge_images(scaled_image, transformed_overlay_image, inpainted_resized, mask_bounds)
        result_image.save(output_path)
        print(f"Image processed and saved to {output_path}")

    def get_mask(self, image, score_threshold=0.8):
        transform = T.Compose([T.ToTensor()])
        image_tensor = transform(image.convert("RGB"))
        with torch.no_grad():
            prediction = self.model([image_tensor])
        masks, scores = prediction[0]['masks'], prediction[0]['scores']
        if len(scores) > 0 and scores.max() >= score_threshold:
            return masks[scores.argmax()].squeeze().numpy()
        return None

    def refine_mask(self, mask):
        mask = (mask * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel) / 255.0

    def adjust_overlay_size(self, overlay_image, base_mask, scale_factor, original_size, target_size):
        orig_width, orig_height = original_size
        target_width, target_height = target_size

        # Calculate mask boundaries
        mask_indices = np.column_stack(np.where(base_mask > 0))
        min_y, min_x = mask_indices.min(axis=0)
        max_y, max_x = mask_indices.max(axis=0)
        mask_width = max_x - min_x
        mask_height = max_y - min_y

        # Calculate scaling factor for the overlay image
        overlay_ratio = overlay_image.width / overlay_image.height
        new_width = int(mask_width * scale_factor * target_width / orig_width)
        new_height = int(new_width / overlay_ratio)

        # Ensure new height does not exceed 1.2 times the mask height
        max_height = int(mask_height * 1.2 * target_height / orig_height)
        if new_height > max_height:
            new_height = max_height
            new_width = int(new_height * overlay_ratio)

        overlay_resized = overlay_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return overlay_resized, (min_x * target_width // orig_width, min_y * target_height // orig_height, max_x * target_width // orig_width, max_y * target_height // orig_height)

    def merge_images(self, base_image, overlay_image, inpainted_image, mask_bounds):
        min_x, min_y, max_x, max_y = mask_bounds
        base_image_np = np.array(base_image)
        overlay_np = np.array(overlay_image.convert("RGBA"))
        inpainted_np = np.array(inpainted_image.convert("RGBA"))

        # Calculate position to place the overlay image such that its bottom aligns with the mask's bottom and is centered horizontally
        overlay_position_y = max_y - overlay_image.height
        overlay_position_x = min_x + (max_x - min_x - overlay_image.width) // 2

        # Keep transparent background
        combined_np = inpainted_np.copy()

        # Ensure the overlay image can extend beyond the mask boundaries
        overlay_top_y = overlay_position_y
        overlay_bottom_y = overlay_position_y + overlay_np.shape[0]
        overlay_left_x = overlay_position_x
        overlay_right_x = overlay_position_x + overlay_np.shape[1]

        combined_top_y = max(0, overlay_top_y)
        combined_bottom_y = min(combined_np.shape[0], overlay_bottom_y)
        combined_left_x = max(0, overlay_left_x)
        combined_right_x = min(combined_np.shape[1], overlay_right_x)

        for y in range(combined_top_y, combined_bottom_y):
            for x in range(combined_left_x, combined_right_x):
                oy = y - overlay_top_y
                ox = x - overlay_left_x
                if overlay_np[oy, ox, 3] > 0:  # Only copy non-transparent parts
                    combined_np[y, x] = overlay_np[oy, ox]

        # Convert combined array to a PIL image
        combined_image = Image.fromarray(combined_np, "RGBA")

        # Apply Gaussian blur to smooth edges
        alpha = combined_image.split()[-1]
        alpha_blurred = alpha.filter(ImageFilter.GaussianBlur(radius=2))
        combined_image.putalpha(alpha_blurred)

        return combined_image

if __name__ == "__main__":
    # Setup directories and paths
    os.makedirs('output_test', exist_ok=True)
    replacer = ImageReplacer(deepfill_model_path, save_intermediate=True)

    base_image_path = r"C:\Users\USER\Desktop\Develop\product-image-generator-backend\static\20240714164305\background.png"
    overlay_image_path = r"C:\Users\USER\Desktop\Develop\product-image-generator-backend\static\product_transparent\20240714164305\product_example_02_transparent.png"

    # Set target sizes, corresponding scale factors, and output paths
    target_sizes_and_factors = [
        ((300, 250), 7.0, 'output_test/replaced_300x250.png'),
        ((320, 480), 7.0, 'output_test/replaced_320x480.png'),
        ((336, 280), 7.0, 'output_test/replaced_336x280.png')
    ]

    for target_size, scale_factor, output_path in target_sizes_and_factors:
        replacer.process_image(base_image_path, overlay_image_path, output_path, target_size, scale_factor)
