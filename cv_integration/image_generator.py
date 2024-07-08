import os
from datetime import datetime
import cv2
import random

import sys
from configs import import_path, static_path
sys.path.append(import_path)
from product_image_processor import ProductImageProcessor
from image_processor import ImageProcessor
from check_product import check_process

# 廣告背景需要選擇用 OPENAI 還是用 SDXL1.0 生成
from openai_generator.openai_generator import translate_text_gpt#, generate_image
from bedrock.text_to_image import text_to_image_request


# 調整圖片大小
def resize_and_pad(image, target_width, target_height):
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
    
    try:
        # 方法一 - OpenAI
        # 建立中文提示
        chinese_prompt = (
            f"產生簡潔清晰的廣告背景，該背景適合從事{interest}。圖像中但沒有任何動物或品牌標誌。"
            f"重點應該是{interest}氛圍且利用三分法則打造{interest}的高質感場地，且不包含出現{product_name}本身與相似的物品。"
            f"確保場景適合{age}歲的{gender}性，且不包含任何特定的文化符號文字。"
            f"此廣告為推廣{product_feature}的{product_name}。"
        )

        # 翻譯成英文
        english_prompt = translate_text_gpt(chinese_prompt)
        print(f"英文提示: {english_prompt}")

        # 開始生成圖片並保存到 background/ 目錄
        print("開始生成圖片並保存到 background/ 目錄")
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        background_path = os.path.join(static_path, "background", f"background_{now}.png")
        generate_image(english_prompt, size="512x512", output_file=background_path)

    except:
        # 方法二 - Bedrock SDXL1.0
        # First combination
        product_name_eng = translate_text_gpt(product_name)
        product_feature_eng = translate_text_gpt(product_feature)
        gender_eng = translate_text_gpt(gender)
        interest_eng = translate_text_gpt(interest)
        user_age_group = "25-34"
        information_level = "detailed"
        content_direction = "healthy living"
        advertising_material_production_direction = "brand awareness"
        brightness = "bright"
        hue = "fresh green"
        saturation = "vibrant"

        # Second combination
        # interest = "outdoor travel"
        # gender = "female"
        # user_age_group = "35-44"
        # information_level = "in-depth"
        # content_direction = "nature conservation"
        # advertising_material_production_direction = "promoting eco-friendly products"
        # brightness = "bright"
        # hue = "forest green"
        # saturation = "saturated"

        MODEL_ID = "stability.stable-diffusion-xl-v1"
        # Define the positive prompt
        POSITIVE_PROMPT = (
            f"The background of ads Photography of {information_level} where to place ads"
            f"{interest}, {gender}, {user_age_group}, {content_direction}, {advertising_material_production_direction}, {brightness}, {hue}, {saturation}, "
            "high resolution, high quality, professional color grading, clear shadows and highlights, atmospheric"
        )

        # Define the negative prompt
        NEGATIVE_PROMPT = (
            "low resolution, bad quality, ugly, blur, mutation, extra arms, extra legs, 3d, painting, "
            "cannot print any main characters (focus solely on the background), the image should not be overcrowded"
        )

        # 開始生成圖片並保存到 background/ 目錄 (TODO: 將名稱一個個翻譯至英文)
        seed = random.randrange(1, 1000000)
        seed = 8070
        text_to_image_request(MODEL_ID, POSITIVE_PROMPT, NEGATIVE_PROMPT, seed, background_path)



    # 開始調整圖像大小並保存到 creative/ 目錄
    print("開始調整圖像大小並保存到 creative/ 目錄")
    img_name = os.path.basename(background_path).split(".")[0]
    output_path_prefix = os.path.join(static_path, "creative")
    output_paths = [f"{output_path_prefix}/creative_{img_name}_{width}x{height}.png" for width, height in sizes]
    resize_image(background_path, sizes, output_paths)

    # 合成最後素材
    print("合成最後素材")
    result_paths = []
    for img, scale in zip(output_paths, [(0.8), (1.2), (1.0)]):
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