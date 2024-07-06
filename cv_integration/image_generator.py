import openai
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
from datetime import datetime
import cv2

import sys
from configs import import_path, static_path, sizes, product_image_path
sys.path.append(import_path)
from product_image_processor import ProductImageProcessor
from image_processor import ImageProcessor
from check_product import check_process

# 加載環境變量
load_dotenv()

# 設置 OpenAI API 密鑰
openai.api_key = os.getenv("API_KEY")

if not openai.api_key:
    raise ValueError("OpenAI API key is not set. Please set the API_KEY environment variable.")

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

def get_product(product_path, static_path, timestamp):
    img_product = os.path.basename(product_path).split(".")[0]
    trans_path_prefix = os.path.join(static_path, "product_transparent", timestamp)
    trans_path = os.path.join(trans_path_prefix, f"{img_product}_transparent.png")

    if not os.path.exists(trans_path):
        print("透明化處理中")
        os.makedirs(trans_path_prefix, exist_ok=True)

        processor = ProductImageProcessor()
        mask, image_rgb = processor.segment_product(product_path)
        result_image = processor.apply_mask(image_rgb, mask)
        result_image.save(trans_path)

        print("透明化完成")
    else:
        print("透明化已完成")

    return trans_path

def get_result(product_name, product_feature, gender, age, job, interest, transparent_path, sizes=[(300, 250), (320, 480), (970, 250)]):
    
    # 建立中文提示
    chinese_prompt = (
        f"為廣告背景打造充滿活力，適合從事{interest}。圖像中但沒有任何動物或品牌標誌。"
        f"重點應該是{interest}氛圍且利用三分法則打造{interest}場地，禁止出現{product_name}本身。"
        f"確保場景適合{age}歲的{gender}性，並且不包含任何特定的文化符號文字。"
        f"此廣告為推廣{product_feature}的{product_name}，可點綴一點與{product_name}相關的元素"
    )

    # 翻譯成英文
    english_prompt = translate_text_gpt(chinese_prompt)
    print(f"英文提示: {english_prompt}")

    # 開始生成圖片並保存到 background/ 目錄
    print("開始生成圖片並保存到 background/ 目錄")
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    background_path = os.path.join(static_path, "background", f"background_{now}.png")
    generate_image(english_prompt, size="512x512", output_file=background_path)

    # 開始調整圖像大小並保存到 creative/ 目錄
    print("開始調整圖像大小並保存到 creative/ 目錄")
    img_name = os.path.basename(background_path).split(".")[0]
    output_path_prefix = os.path.join(static_path, "creative")
    output_paths = [f"{output_path_prefix}/creative_{img_name}_{width}x{height}.png" for width, height in sizes]
    resize_image(background_path, sizes, output_paths)

    # 合成最後素材
    print("合成最後素材")
    result_paths = []
    for img, scale in zip(output_paths, [0.8, 1.2, 1.0]):
        print(f"處理圖片: {background_path}")
        result_path_prefix = os.path.join(static_path, "result")
        result_path = f"{result_path_prefix}/result_{os.path.basename(img)}"
        
        # 若圖片中包含產品，則利用產品取代之 (例如 water bottle → water, bottle)
        product_name_english = translate_text_gpt(product_name)
        prossible_object = product_name_english.split(" ")
        for keyword in prossible_object:
            result = check_process(target_label=keyword, background_path=img, target_image_path=transparent_path, output_image_path=result_path, scale=scale*1.2)
            if result:
                break

        # 若圖片中未包含產品，則找最佳位置放置
        if result:
            print("圖片中包含產品，且處理完成。")
        else:
            print("圖片中未包含產品，正在處理。")
            processor = ImageProcessor(transparent_path, img)
            processor.composite_images(scale=scale)
            processor.enhance_foreground_saturation()
            processor.save_image(result_path)
            print(f"圖片保存為: {result_path}")

        result_paths.append(result_path)

    return result_paths

if __name__ == "__main__":

    # 設置基本信息
    product_name = "水瓶"
    product_feature = "方便攜帶"
    gender = "男性"
    age = "25-34"
    job = "軟體工程師"
    interest = "運動體育"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 取得透明化圖片的路徑
    transparent_path = get_product(product_image_path, static_path, timestamp)
    
    # 生成結果
    result_paths = get_result(product_name, product_feature, gender, age, job, interest, transparent_path, sizes=sizes)
    
    # 打印結果路徑
    print("生成的圖片路徑:")
    for path in result_paths:
        print(path)