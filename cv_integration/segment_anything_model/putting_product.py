import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
import torchvision.transforms as T
from segment_anything import SamPredictor, sam_model_registry

# Set system paths
import sys
sys.path.append("./../../")  # Assuming other necessary modules are located here
from configs import model_type, checkpoint_path, deepfill_model_path, import_path
sys.path.append(import_path)

# Define the SegmentAnythingProcessor for image segmentation
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
        self.mask_bounds = self.calculate_mask_bounds(self.best_mask)
        return best_mask

    def save_masks(self, original_bg_path):
        mask_image_original = self.image.copy()
        mask_image_original_np = np.array(mask_image_original, dtype=np.uint8)
        mask = (self.best_mask * 255).astype(np.uint8)
        mask_3channel = np.stack([mask, mask, mask], axis=-1)
        mask_image_original_np = np.where(mask_3channel == 0, mask_image_original_np, [255, 255, 255])
        mask_image_original = Image.fromarray(mask_image_original_np.astype(np.uint8))
        mask_image_original.save(original_bg_path)

    def calculate_mask_bounds(self, mask):
        mask_indices = np.argwhere(mask)
        y_min, x_min = mask_indices.min(axis=0)
        y_max, x_max = mask_indices.max(axis=0)
        return x_min, y_min, x_max, y_max

# Define ImageCompositor for handling image compositing and inpainting
class ImageCompositor:
    def __init__(self, processor, deepfill_model_path):
        self.processor = processor
        self.deepfill = DeepFill(deepfill_model_path)

    def load_and_process(self, image_path, mask):
        self.processor.set_image(image_path)
        self.processor.segment_center_object()
        self.processor.save_masks(mask)

    def composite_images(self, base_image_path, overlay_image_path, output_path, transparent_bg_path):
        base_image = Image.open(base_image_path).convert("RGBA")
        overlay_image = Image.open(overlay_image_path).convert("RGBA")
        
        # Use the mask bounds from the processor
        mask_bounds = self.processor.mask_bounds
        mask_height = mask_bounds[3] - mask_bounds[1]
        
        # Resize the overlay image to match the height of the mask area while maintaining aspect ratio
        overlay_width, overlay_height = overlay_image.size
        scale_factor = mask_height / overlay_height
        new_width = int(overlay_width * scale_factor)
        overlay_image_resized = overlay_image.resize((new_width, mask_height), Image.Resampling.LANCZOS)
        
        # Create a new image for the composite result
        result_image = Image.new("RGBA", base_image.size)
        result_image.paste(base_image, (0, 0))
        
        # Calculate the horizontal center position for overlay image based on the mask bounds
        horizontal_center = (mask_bounds[0] + mask_bounds[2]) // 2
        paste_x = horizontal_center - new_width // 2
        paste_y = mask_bounds[1]
        
        # Paste the resized overlay image into the mask region, centered horizontally
        result_image.paste(overlay_image_resized, (paste_x, paste_y), overlay_image_resized)
        
        # Save the composite image
        result_image.save(output_path)
        
        # Generate the mask for the composite image
        composite_mask = self.generate_composite_mask(result_image)
        
        # Smooth the edges of the mask
        smooth_mask = self.smooth_edges(composite_mask)
        
        # Save the smoothed mask as a transparent mask
        self.save_transparent_mask(smooth_mask, transparent_bg_path)
        
        # Use DeepFill to inpaint the composite image using the smoothed mask
        self.deepfill.inpaint(output_path, transparent_bg_path, output_path)

    def generate_composite_mask(self, composite_image):
        composite_np = np.array(composite_image)
        gray = cv2.cvtColor(composite_np, cv2.COLOR_RGBA2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        return binary / 255.0

    def smooth_edges(self, mask, blur_radius=5):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img_blurred = mask_img.filter(ImageFilter.GaussianBlur(blur_radius))
        smooth_mask = np.array(mask_img_blurred) / 255.0
        return smooth_mask

    def save_transparent_mask(self, mask, transparent_bg_path):
        mask = (mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask)
        mask_rgba = Image.new("RGBA", mask_image.size)
        mask_rgba.paste(mask_image, mask=(mask_image))
        mask_rgba.save(transparent_bg_path)
        print(f"Mask saved to {transparent_bg_path}")

# Define DeepFill for inpainting
class DeepFill:
    def __init__(self, model_path, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        generator_state_dict = torch.load(model_path, map_location=self.device)['G']
        if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
            from deepfill_v2_model.model.networks import Generator
        else:
            from deepfill_v2_model.model.networks_tf import Generator
        self.generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(self.device)
        self.generator.load_state_dict(generator_state_dict, strict=True)
        self.generator.eval()

    def inpaint(self, image_path, mask_path, output_path):
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)
        _, h, w = image.shape
        grid = 8
        image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
        mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)
        image = (image * 2 - 1.).to(self.device)
        mask = (mask > 0.5).to(dtype=torch.float32, device=self.device)
        image_masked = image * (1. - mask)
        ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
        x = torch.cat([image_masked, ones_x, ones_x * mask], dim=1)
        with torch.inference_mode():
            _, x_stage2 = self.generator(x, mask)
        image_inpainted = image * (1. - mask) + x_stage2 * mask
        img_out = ((image_inpainted[0].permute(1, 2, 0) + 1) * 127.5).to(device='cpu', dtype=torch.uint8)
        img_out = Image.fromarray(img_out.numpy())
        img_out.save(output_path)
        print(f"Saved output file at: {output_path}")

if __name__ == "__main__":
    processor = SegmentAnythingProcessor()
    compositor = ImageCompositor(processor, deepfill_model_path)

    mask = "original_bg_mask.png"
    masked = "transparent_composite_image.png"

    print("取得背景遮罩")
    test1 = r"C:\Users\USER\Desktop\Develop\sd_sagemaker\content\generated_images\text-image\generated_image_20240710031944.png"
    compositor.load_and_process(test1, mask=mask)

    print("合成图像")
    compositor.composite_images(mask, r"C:\Users\USER\Desktop\Develop\product-image-generator-backend\cv_integration\product_test\product_example_01_trans.png", "composited_image.png", masked)
