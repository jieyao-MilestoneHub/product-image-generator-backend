import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T
import os

# Set system paths
import sys
sys.path.append("./../../")  # Assuming other necessary modules are located here
from configs import deepfill_model_path, import_path
sys.path.append(import_path)

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

    def inpaint(self, image, mask):
        image = T.ToTensor()(image).unsqueeze(0)
        mask = T.ToTensor()(mask).unsqueeze(0)
        _, h, w = image.shape[1:]
        grid = 8
        image = image[:, :3, :h // grid * grid, :w // grid * grid]
        mask = mask[:, :1, :h // grid * grid, :w // grid * grid]
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
        return img_out

class ImageReplacer:
    def __init__(self, deepfill_model_path, save_intermediate=False):
        # 加载 Mask R-CNN 模型
        self.model = maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()
        self.deepfill = DeepFill(deepfill_model_path)
        self.save_intermediate = save_intermediate

    def get_mask(self, image, score_threshold=0.8):
        # 将图像转换为张量
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        image_rgb = image.convert("RGB")  # 确保图像是 RGB 格式
        image_tensor = transform(image_rgb)
        with torch.no_grad():
            prediction = self.model([image_tensor])
        masks = prediction[0]['masks']
        scores = prediction[0]['scores']

        # 选择高于阈值的最高置信度的掩码
        if len(scores) > 0:
            best_score_idx = scores.argmax()
            if scores[best_score_idx] >= score_threshold:
                best_mask = masks[best_score_idx].squeeze().numpy()
                return best_mask
        return None

    def refine_mask(self, mask):
        # 转换为二值图像
        mask = (mask * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 应用形态学操作（腐蚀然后膨胀）
        kernel = np.ones((5, 5), np.uint8)
        refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # 返回浮点型掩码
        return refined_mask / 255.0

    def replace_object(self, base_image_path, overlay_image_path):
        # 读取图像
        base_image = Image.open(base_image_path).convert("RGBA")
        overlay_image = Image.open(overlay_image_path).convert("RGBA")

        # 获取图像 B 中的物体掩码
        base_mask = self.get_mask(base_image)
        if base_mask is None:
            print("未能检测到物体")
            return None
        base_mask = self.refine_mask(base_mask)

        # 获取图像 A 中的物体掩码
        overlay_mask = self.get_mask(overlay_image)
        if overlay_mask is None:
            print("未能检测到覆盖物体")
            return None
        overlay_mask = self.refine_mask(overlay_mask)

        # 计算放置位置
        base_width, base_height = base_image.size
        overlay_width, overlay_height = overlay_image.size

        # 获取目标位置
        base_mask_indices = np.column_stack(np.where(base_mask > 0))
        overlay_mask_indices = np.column_stack(np.where(overlay_mask > 0))

        if base_mask_indices.size == 0 or overlay_mask_indices.size == 0:
            print("无法检测到有效的掩码区域")
            return None

        # 计算透视变换矩阵
        base_pts = np.float32([
            [base_mask_indices[:, 1].min(), base_mask_indices[:, 0].min()],
            [base_mask_indices[:, 1].max(), base_mask_indices[:, 0].min()],
            [base_mask_indices[:, 1].max(), base_mask_indices[:, 0].max()],
            [base_mask_indices[:, 1].min(), base_mask_indices[:, 0].max()]
        ])

        overlay_pts = np.float32([
            [overlay_mask_indices[:, 1].min(), overlay_mask_indices[:, 0].min()],
            [overlay_mask_indices[:, 1].max(), overlay_mask_indices[:, 0].min()],
            [overlay_mask_indices[:, 1].max(), overlay_mask_indices[:, 0].max()],
            [overlay_mask_indices[:, 1].min(), overlay_mask_indices[:, 0].max()]
        ])

        M = cv2.getPerspectiveTransform(overlay_pts, base_pts)

        # 透视变换覆盖图像（包括透明度通道）
        overlay_image_np = np.array(overlay_image)
        overlay_image_transformed = cv2.warpPerspective(
            overlay_image_np, M, (base_width, base_height), borderMode=cv2.BORDER_TRANSPARENT)

        # 创建遮罩
        overlay_mask_transformed = cv2.warpPerspective(
            overlay_mask, M, (base_width, base_height), borderMode=cv2.BORDER_TRANSPARENT)

        # 准备修补图像和透明背景的白色遮罩
        base_image_np = np.array(base_image)
        inpaint_image_np = base_image_np.copy()
        inpaint_image_np[overlay_mask_transformed > 0] = [255, 255, 255, 255]  # 将遮罩区域设为白色

        inpaint_image_pil = Image.fromarray(inpaint_image_np)
        overlay_mask_pil = Image.fromarray((overlay_mask_transformed * 255).astype(np.uint8)).convert("L")

        # 保存遮罩和修补图像（仅在 save_intermediate 为 True 时保存）
        if self.save_intermediate:
            mask_output_path = os.path.join('mask', f'{os.path.basename(base_image_path)}_mask.png')
            inpaint_output_path = os.path.join('masked', f'{os.path.basename(base_image_path)}_masked.png')
            overlay_mask_pil.save(mask_output_path)
            inpaint_image_pil.save(inpaint_output_path)

        # 使用 DeepFill 修补目标区域
        inpainted_image = self.deepfill.inpaint(inpaint_image_pil, overlay_mask_pil)

        # 将 overlay 图像的 alpha 通道分离出来
        alpha_channel = overlay_image_transformed[:, :, 3] / 255.0
        overlay_rgb = overlay_image_transformed[:, :, :3]

        # 将修补后的图像和 overlay 图像进行融合
        inpainted_image_np = np.array(inpainted_image)
        inpainted_rgb = inpainted_image_np[:, :, :3]
        # 确保修补后的图像具有 alpha 通道
        inpainted_alpha = np.ones((inpainted_rgb.shape[0], inpainted_rgb.shape[1]))

        combined_rgb = (overlay_rgb * alpha_channel[..., None] + inpainted_rgb * (1 - alpha_channel[..., None])).astype(np.uint8)
        combined_alpha = (alpha_channel + inpainted_alpha * (1 - alpha_channel)).astype(np.uint8) * 255

        combined_image_np = np.dstack([combined_rgb, combined_alpha])

        # 返回最终结果图像
        result_image = Image.fromarray(combined_image_np, "RGBA")
        return result_image

    def save_image(self, image, path):
        image.save(path)

def resize_and_pad(image, target_width, target_height, pad_color=(0, 0, 0)):
    """
    Resize and pad an image to the target width and height.

    Parameters:
    - image: Input image as a numpy array.
    - target_width: Desired width of the output image.
    - target_height: Desired height of the output image.
    - pad_color: Color to use for padding (default is black).

    Returns:
    - new_image: The resized and padded image as a numpy array.
    """
    original_height, original_width = image.shape[:2]
    original_channels = image.shape[2] if len(image.shape) > 2 else 1

    # Calculate scaling factor to fit image within target dimensions
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize the image with high quality
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Calculate padding
    delta_w = target_width - new_width
    delta_h = target_height - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Pad the image with the specified color
    if original_channels == 1:  # Grayscale image
        pad_color = pad_color[0]  # Use a single value for grayscale padding
    new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)

    return new_image

if __name__ == "__main__":
    # 创建保存目录
    os.makedirs('mask', exist_ok=True)
    os.makedirs('masked', exist_ok=True)
    os.makedirs('inpainted', exist_ok=True)
    os.makedirs('test_image_result', exist_ok=True)

    replacer = ImageReplacer(deepfill_model_path)

    base_image_path = r"C:\Users\USER\Desktop\Develop\sd_sagemaker\content\generated_images\text-image\generated_image_20240710143809.png"
    overlay_image_path = r"C:\Users\USER\Desktop\Develop\product-image-generator-backend\cv_integration\product_test\product_example_02_trans.png"
    
    # 处理图像
    result_image = replacer.replace_object(base_image_path, overlay_image_path)
    if result_image is not None:
        output_paths = [
            "test_image_result/replaced_300x250.png",
            "test_image_result/replaced_320x480.png",
            "test_image_result/replaced_336x280.png"
        ]
        sizes = [(300, 250), (320, 480), (336, 280)]

        for size, output_path in zip(sizes, output_paths):
            resized_image = result_image.resize(size, Image.Resampling.LANCZOS)
            replacer.save_image(resized_image, output_path)
            print(f"Saved resized image to {output_path}")
