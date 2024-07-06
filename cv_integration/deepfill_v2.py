import torch
import torchvision.transforms as T
from PIL import Image

class DeepFill:
    def __init__(self, model_path, device=None):
        # 检查设备，如果未指定设备，默认使用CUDA，如果不可用则使用CPU
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # 加载模型状态字典
        generator_state_dict = torch.load(model_path, map_location=self.device)['G']
        
        # 检查权重文件格式，选择相应的网络定义文件
        if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
            from deepfill_v2_model.model.networks import Generator
        else:
            from deepfill_v2_model.model.networks_tf import Generator

        # 初始化生成器
        self.generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(self.device)
        self.generator.load_state_dict(generator_state_dict, strict=True)
        self.generator.eval()

    def inpaint(self, image_path, mask_path, output_path):
        # 加载图像和掩码
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # 转换为张量
        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)

        _, h, w = image.shape
        grid = 8

        image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
        mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

        print(f"Shape of image: {image.shape}")

        image = (image * 2 - 1.).to(self.device)  # 将图像值映射到 [-1, 1] 范围
        mask = (mask > 0.5).to(dtype=torch.float32, device=self.device)  # 1.: 被掩盖的，0.: 未掩盖的

        image_masked = image * (1. - mask)  # 掩盖图像

        ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
        x = torch.cat([image_masked, ones_x, ones_x * mask], dim=1)  # 连接通道

        with torch.inference_mode():
            _, x_stage2 = self.generator(x, mask)

        # 完成图像
        image_inpainted = image * (1. - mask) + x_stage2 * mask

        # 保存修复后的图像
        img_out = ((image_inpainted[0].permute(1, 2, 0) + 1) * 127.5).to(device='cpu', dtype=torch.uint8)
        img_out = Image.fromarray(img_out.numpy())
        img_out.save(output_path)

        print(f"Saved output file at: {output_path}")

if __name__ == '__main__':

    model_path = 'deepfill_v2_model/pretrained/states_tf_places2.pth'

    # 加载DeepFill模型
    deepfill = DeepFill(model_path)

    # 加载测试图像和掩码
    image_path = 'deepfill_v2_model/examples/inpaint/case1.png'
    mask_path = 'deepfill_v2_model/examples/inpaint/case1_mask.png'
    output_path = 'case1_out_test.png'

    # 进行图像修复
    deepfill.inpaint(image_path, mask_path, output_path)
