import openai
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
from datetime import datetime
from raw_trans import ProductImageProcessor
from trans_comp import ImageProcessor

from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key = os.getenv("OPENAI_API")

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

def generate_image(prompt, size, output_file):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size=size
    )
    image_url = response['data'][0]['url']

    # 下載圖片並保存
    image_response = requests.get(image_url)
    img = Image.open(BytesIO(image_response.content))
    img.save(output_file)
    print(f"圖片已保存為 {output_file}")

def generate_and_save_image(product_name, product_feature, gender, age, job, interest):
    print("利用 API 生成圖片中")

    # 建立中文提示
    # chinese_prompt = (
    #     f"產生一張對於{age}歲的{gender}的產品廣告背景圖。"
    #     f"請專注於{interest}的地點，背景必須具有代表性，例如對於喜歡{interest}的人，他們會在該場景進行{interest}。"
    #     f"圖片中不能包含{product_name}。"
    #     f"此廣告需營造{product_feature}的感覺，帶入真實感。"
    #     f"影像風格要求明確，以突顯{interest}的使用場景。"
    # )

    chinese_prompt = (
        f"產生一張廣告背景圖，針對{age}歲的{gender}性進行銷售。"
        f"專注於產生{interest}的地點，背景必須具有代表性，例如對於喜歡{interest}的人，他們會在該場景進行{interest}。"
        f"圖片中不能包含{product_name}。"
        f"此廣告需營造{product_feature}。"
        f"影像風格要求明確，以突顯{interest}的場景。"
    )

    # 翻譯成英文
    english_prompt = translate_text_gpt(chinese_prompt)
    print(f"英文提示: {english_prompt}")

    # 生成圖片
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    generate_image(english_prompt, size="512x512", output_file=f"background_{now}.png")


def extract_product(tmp_image_trans_path="product_transparent.png"):
    print("擷取產品圖")
    processor = ProductImageProcessor()
    mask, image_rgb = processor.segment_product('product.png')
    result_image = processor.apply_mask(image_rgb, mask)
    result_image.save(tmp_image_trans_path)


def composition(background_img_path, result_path, tmp_image_trans_path="product_transparent.png"):
    print("圖片合成")
    processor = ImageProcessor(tmp_image_trans_path, background_img_path)
    processor.composite_images(scale=1.2)  # 縮放產品圖並合成
    processor.enhance_foreground_saturation()
    processor.save_image(result_path)
    print(f"圖片保存為: {result_path}")
    if os.path.exists(tmp_image_trans_path):
        os.remove(tmp_image_trans_path)
        print(f"文件 {tmp_image_trans_path} 已被删除。")
    else:
        print(f"文件 {tmp_image_trans_path} 不存在。")


def main():
    product_name = "水壺"
    product_feature = "方便攜帶"
    gender = "男"
    age = "25~34"
    job = "軟體工程師"
    interest = "運動、健身及休閒娛樂指導員"
    generate_and_save_image(product_name, product_feature, gender, age, job, interest)
    extract_product()
    background_img_path = ""
    result_path = ""
    composition(background_img_path, result_path)