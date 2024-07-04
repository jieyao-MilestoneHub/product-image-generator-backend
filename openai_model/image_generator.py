import openai
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
from datetime import datetime
import cv2
from product_image_processor import ProductImageProcessor
from image_processor import ImageProcessor

# 加載環境變量
load_dotenv()

# 設置 OpenAI API 密鑰
openai.api_key = os.getenv("API_KEY")

# 使用 GPT 模型進行文本翻譯
def translate_text_gpt(text, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates Chinese to English."},
            {"role": "user", "content": f"Translate the following Chinese text to English accurately and completely:\n\n{text}"}
        ],
        max_tokens=200,
        n=1,
        temperature=0.5,
    )
    translation = response['choices'][0]['message']['content'].strip()
    return translation

# 使用 OpenAI API 生成圖片
def generate_image(prompt, size, output_file):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size=size
    )
    image_url = response['data'][0]['url']

    # 下載並保存圖片
    image_response = requests.get(image_url)
    img = Image.open(BytesIO(image_response.content))
    img.save(output_file)
    print(f"圖片已保存為 {output_file}")

# 調整圖片大小
def resize_and_pad(image, target_width, target_height, background_color=(255, 255, 255)):
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def resize_image(image_path, sizes, output_paths):
    image = cv2.imread(image_path)
    for (width, height), output_path in zip(sizes, output_paths):
        resized_image = resize_and_pad(image, width, height)
        cv2.imwrite(output_path, resized_image)

def main():
    product_name = "水壺"
    product_feature = "方便攜帶"
    gender = "男"
    age = "25~34"
    job = "軟體工程師"
    interest = "運動、健身及休閒娛樂指導員"

    # 建立中文提示
    chinese_prompt = (
        f"為廣告背景打造充滿活力，適合從事{interest}。圖像中但沒有任何動物或品牌標誌。"
        f"重點應該是{interest}氛圍且利用三分法則打造{interest}場地，保留一個{product_name}大小的空白且不包含{product_name}本身。"
        f"確保場景適合{age}歲的{gender}性，並且不包含任何特定的文化符號文字。"
        f"此廣告為推廣{product_feature}的{product_name}，可點綴一點與{product_name}相關的元素"
    )

    # 翻譯成英文
    english_prompt = translate_text_gpt(chinese_prompt)
    print(f"英文提示: {english_prompt}")

    # 生成圖片並保存到 ./background/ 目錄
    print("生成圖片並保存到 ./background/ 目錄")
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    background_path = f"./background/background_{now}.png"
    generate_image(english_prompt, size="512x512", output_file=background_path)

    # 調整圖像大小並保存到 ./creative/ 目錄
    print("調整圖像大小並保存到 ./creative/ 目錄")
    img_name = os.path.basename(background_path).split(".")[0]
    sizes = [(300, 250), (320, 480)]
    output_paths = [f"./creative/creative_{img_name}_{sizes[0][0]}x{sizes[0][1]}.png", f"./creative/creative_{img_name}_{sizes[1][0]}x{sizes[1][1]}.png"]
    resize_image(background_path, sizes, output_paths)

    # 處理產品圖像
    print("處理產品圖像")
    processor = ProductImageProcessor()
    mask, image_rgb = processor.segment_product('./product/product_example_01.png')
    result_image = processor.apply_mask(image_rgb, mask)
    result_image.save('./product/product_transparent.png')

    # 合成最後素材
    print("合成最後素材")
    for img, scale in zip(output_paths, [0.8, 1.2]):
        print(f"處理圖片: {background_path}")
        processor = ImageProcessor('./product/product_transparent.png', img)
        processor.composite_images(scale=scale)
        processor.enhance_foreground_saturation()
        result_path = f"./result/result_{os.path.basename(img)}"
        processor.save_image(result_path)
        print(f"圖片保存為: {result_path}")

if __name__ == "__main__":
    main()
