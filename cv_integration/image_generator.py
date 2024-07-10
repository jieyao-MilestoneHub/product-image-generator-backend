import os
from datetime import datetime
import cv2
import random
from PIL import Image

import sys
from configs import import_path, static_path, deepfill_model_path
sys.path.append(import_path)
from product_image_processor import ProductImageProcessor

# 廣告背景需要選擇用 OPENAI 還是用 SDXL1.0 生成
from openai_generator.openai_generator import translate_text_gpt
from bedrock.text_to_image import text_to_image_request
from mask_cnn_model.putting_product import ImageReplacer


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
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    static_path = 'path_to_static'  # 需要替换为实际的静态路径
    background_path = os.path.join(static_path, "background", f"background_{now}.png")

    product_name_eng = translate_text_gpt(product_name)
    product_feature_eng = translate_text_gpt(product_feature)
    gender_eng = translate_text_gpt(gender)
    interest_eng = translate_text_gpt(interest)
    try_time = 0

    while try_time < 5:
        try:
            MODEL_ID = "stability.stable-diffusion-xl-v1"

            POSITIVE_PROMPT = (
                f"ads background for unique {product_name_eng}, ellipse border."
                f"{product_feature_eng}"
                f"focus on visually striking, lifelike advertisement scene for {interest_eng}"
                f"targeted at {age}-year-old {gender_eng}. "
                f"Use dynamic lighting to enhance depth and showcase a minimalistic texture with elements"
                f"Elegantly incorporate explicit {product_name_eng}, ensuring practical use, while focusing on the container's design and functionality rather than the product itself. "
                f"Design the setting with bold, clear lines and a monochromatic color scheme in the left two-thirds of the image, adding decorative elements at the edges to enhance the visual appeal."
                f"masterpiece, high resolution, high quality, hdr, fujifilm xt4, 50mm, f/1.6, sharp focus, high detailed."
            )

            NEGATIVE_PROMPT = (
                f"Do not block {product_name_eng} for any element in image."
                "No human figures, animals, brand logos, or man-made objects should be visible. "
                "Avoid low resolution, blurring, and any distortions to ensure clarity and photographic realism. "
                "Exclude painting-like effects to maintain photographic realism. "
                "Do not include human figures, portraits, or man-made objects. "
                "Avoid placing any elements in the center of the image."
                "low resolution, bed quality, ugly, flur, (mutation), extra arms, extra legs, 3d, painting."
            )

            seed = random.randrange(1, 1000000)
            text_to_image_request(MODEL_ID, POSITIVE_PROMPT, NEGATIVE_PROMPT, seed, background_path)

            replacer = ImageReplacer(deepfill_model_path)
            result_image = replacer.replace_object(background_path, transparent_path)

            if result_image is not None:
                img_name = os.path.basename(background_path).split(".")[0]
                output_path_prefix = os.path.join(static_path, "creative")
                output_paths = [f"{output_path_prefix}/creative_{img_name}_{width}x{height}.png" for width, height in sizes]

                for size, output_path in zip(sizes, output_paths):
                    resized_image = result_image.resize(size, Image.Resampling.LANCZOS)
                    replacer.save_image(resized_image, output_path)
                    print(f"Saved resized image to {output_path}")
                
                return output_paths

            else:
                raise ValueError("Result image is None.")

        except Exception as e:
            try_time += 1
            logging.error(f"Attempt {try_time}: Error occurred - {e}")

    logging.error("Max attempts reached. Failed to generate the desired result.")
    return None