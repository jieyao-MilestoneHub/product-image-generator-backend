import torch
import torchvision.transforms as T
from PIL import Image

import sys
sys.path.append("./../")
from configs import deepfill_model_path

class DeepFill:
    def __init__(self, model_path, device=None):
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # 載入模型狀態字典
        generator_state_dict = torch.load(model_path, map_location=self.device)['G']
        
        # 檢察權重文件格式，選擇對應的網路文件
        if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
            from deepfill_v2_model.model.networks import Generator
        else:
            from deepfill_v2_model.model.networks_tf import Generator

        # 初始化生成器
        self.generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(self.device)
        self.generator.load_state_dict(generator_state_dict, strict=True)
        self.generator.eval()

    def inpaint(self, image_path, mask_path, output_path):
        # 載入圖像及遮罩
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # 轉換為張量
        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)

        _, h, w = image.shape
        grid = 8

        image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
        mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

        print(f"Shape of image: {image.shape}")

        image = (image * 2 - 1.).to(self.device)  # 將圖像映射到 [-1, 1]
        mask = (mask > 0.5).to(dtype=torch.float32, device=self.device)  # 1.: 有遮罩，0.: 無遮罩

        image_masked = image * (1. - mask)  # 掩盖图像

        ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
        x = torch.cat([image_masked, ones_x, ones_x * mask], dim=1)  # 連接通道

        with torch.inference_mode():
            _, x_stage2 = self.generator(x, mask)

        # 完成圖像
        image_inpainted = image * (1. - mask) + x_stage2 * mask

        # 儲存修復後的圖像
        img_out = ((image_inpainted[0].permute(1, 2, 0) + 1) * 127.5).to(device='cpu', dtype=torch.uint8)
        img_out = Image.fromarray(img_out.numpy())
        img_out.save(output_path)

        print(f"Saved output file at: {output_path}")

if __name__ == '__main__':

    # 載入DeepFill
    deepfill = DeepFill(deepfill_model_path)

    # 測試圖像及遮罩
    image_path = r'C:\Users\USER\Desktop\Develop\product-image-generator-backend\cv_integration\segment_anything_model\composited_image.png'
    mask_path = r'C:\Users\USER\Desktop\Develop\product-image-generator-backend\cv_integration\segment_anything_model\transparent_composite_image.png'
    output_path = 'case_out_test.png'

    # 進行圖像修補
    deepfill.inpaint(image_path, mask_path, output_path)
