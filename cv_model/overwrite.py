from PIL import Image
from datetime import datetime

def overlay_product_on_background(product_image_path, background_image_path, output_image_path, scale_factor=0.9):
    # 打開產品圖片和背景圖片
    product_img = Image.open(product_image_path).convert("RGBA")
    background_img = Image.open(background_image_path).convert("RGBA")

    # 計算產品圖片的新大小
    bg_width, bg_height = background_img.size
    product_aspect = product_img.width / product_img.height
    bg_aspect = bg_width / bg_height

    if product_aspect > bg_aspect:
        new_product_width = int(bg_width * scale_factor)
        new_product_height = int(new_product_width / product_aspect)
    else:
        new_product_height = int(bg_height * scale_factor)
        new_product_width = int(new_product_height * product_aspect)

    # 在保持長寬比的情況下調整產品圖片大小
    product_img = product_img.resize((new_product_width, new_product_height), resample=Image.LANCZOS)

    # 將白色區域設為透明
    product_img = make_white_transparent(product_img)

    # 計算將產品圖片居中放置在背景圖片上的位置
    x = (bg_width - new_product_width) // 2
    y = (bg_height - new_product_height) // 2

    # 創建一個透明層以放置產品圖片
    transparent_layer = Image.new('RGBA', background_img.size, (0, 0, 0, 0))
    transparent_layer.paste(product_img, (x, y), product_img)

    # 將產品圖片合成到背景圖片上
    final_img = Image.alpha_composite(background_img, transparent_layer)

    # 保存最終圖片
    final_img.save(output_image_path)

def make_white_transparent(img):
    # 將圖片轉換為 RGBA 格式
    img = img.convert("RGBA")

    # 獲取圖片的像素數據
    pixdata = img.load()

    # 將白色區域（255, 255, 255）設為透明
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            if pixdata[x, y][:3] == (255, 255, 255):  # 忽略 alpha 通道
                pixdata[x, y] = (255, 255, 255, 0)

    return img


if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    overlay_product_on_background('test_product.jpg', r'C:\Users\USER\Desktop\Develop\product-image-generator-backend\openai_model\ad_image_20240629203812.png', f'output_image_{current_time}.png')
