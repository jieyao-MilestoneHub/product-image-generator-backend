from deepfill_v2 import DeepFill

def main():
    # 模型路径
    model_path = 'pretrained/states_tf_places2.pth'

    # 加载DeepFill模型
    deepfill = DeepFill(model_path)

    # 加载测试图像和掩码
    image_path = 'examples/inpaint/case1.png'
    mask_path = 'examples/inpaint/case1_mask.png'
    output_path = 'examples/inpaint/case1_out_test.png'

    # 进行图像修复
    deepfill.inpaint(image_path, mask_path, output_path)

if __name__ == '__main__':
    main()
