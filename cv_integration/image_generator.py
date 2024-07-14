import os
from datetime import datetime
import shutil
import random
import logging

import sys
sys.path.append("./../")
from configs import import_path, static_path, deepfill_model_path, sizes
sys.path.append(import_path)
from product_image_processor import ProductImageProcessor

# Advertisement background generation using SDXL1.0
from cv_integration.transfer_openai import translate_text_gpt
from cv_integration.text_to_image_bedrock import text_to_image_request
from cv_integration.replace_product import ImageReplacer

# Configure logging to use the same configuration as the main application
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler('app.log', mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# Function to process the product image and make the background transparent
def get_product(product_path, static_path, timestamp):
    img_product = os.path.basename(product_path).split(".")[0]
    trans_path_prefix = os.path.join(static_path, "product_transparent", timestamp)
    trans_path = os.path.join(trans_path_prefix, f"{img_product}_transparent.png")

    if not os.path.exists(trans_path):
        logging.info("Processing transparency")
        os.makedirs(trans_path_prefix, exist_ok=True)

        processor = ProductImageProcessor()
        mask, image_rgb = processor.segment_product(product_path)
        result_image = processor.apply_mask(image_rgb, mask)
        result_image.save(trans_path)

        logging.info("Transparency processing complete")
    else:
        logging.info("Transparency already processed")

    return trans_path

# Function to generate the final result images
def get_result(product_name, product_feature, gender, age, job, interest, transparent_path, generated_path, background_path):

    # Translate text to English using GPT
    product_name_eng = translate_text_gpt(product_name)
    # product_feature_eng = translate_text_gpt(product_feature)
    gender_eng = translate_text_gpt(gender)
    interest_eng = translate_text_gpt(interest)
    try_time = 0

    while try_time < 5:
        try:
            MODEL_ID = "stability.stable-diffusion-xl-v1"

            POSITIVE_PROMPT = (
                f"ads background for unique {product_name_eng} of ellipse border and thin width and short."
                f"focus on visually striking, lifelike advertisement scene for {interest_eng}."
                f"targeted at {age}-year-old {gender_eng}. "
                "Use dynamic lighting to enhance depth and showcase a minimalistic texture with elements."
                f"Elegantly incorporate {product_name_eng}, ensuring practical use, while focusing on the container's design and functionality rather than the product itself. "
                "Design the setting with bold, clear lines and a monochromatic color scheme in the left two-thirds of the image, adding decorative elements at the edges to enhance the visual appeal."
                "natural, masterpiece, high resolution, high quality, hdr, fujifilm xt4, 50mm, f/1.6, sharp focus, high detailed."
            )

            NEGATIVE_PROMPT = (
                f"Do not block any part of {product_name_eng}."
                "No borders."
                "No human figures, animals, brand logos, or man-made objects should be visible. "
                "Avoid low resolution, blurring, and any distortions to ensure clarity and photographic realism. "
                "Exclude painting-like effects to maintain photographic realism. "
                "low resolution, bed quality, ugly, flur, (mutation), extra arms, extra legs, extra fingers, 3d, painting."
            )

            seed = random.randrange(15000, 30000)
            logger.info(seed)
            text_to_image_request(MODEL_ID, POSITIVE_PROMPT, NEGATIVE_PROMPT, seed, background_path)

            # 合成圖片
            replacer = ImageReplacer(deepfill_model_path)
            img_base_name = os.path.basename(background_path).split(".")[0]
            
            # 設置目標尺寸和對應的縮放比例
            target_sizes_and_factors = [
                ((sizes[0][0], sizes[0][1]), 100),
                ((sizes[1][0], sizes[1][1]), 100),
                ((sizes[2][0], sizes[2][1]), 100)
            ]

            output_paths = []
            for target_size, scale_factor in target_sizes_and_factors:

                width, height = target_size
                img_name = f"creative_{img_base_name}_{width}x{height}.png"
                output_path = os.path.join(generated_path, img_name)
                replacer.process_image(background_path, transparent_path, output_path, target_size, scale_factor)
                logging.info(f"Saved resized image to {output_path}")

                # 確保後端正確傳遞圖片
                show_path = generated_path.split(os.sep)
                show_path = os.sep.join(show_path[7:])
                output_paths.append(os.path.join(show_path, img_name))

            return output_paths

        except Exception as e:
            clear_folder(generated_path)
            try_time += 1
            logging.error(f"Attempt {try_time}: Error occurred - {e} (result from produce 'target product')")

    logging.error("Max attempts reached. Failed to generate the desired result.")
    return None

def clear_folder(folder_path):
    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 遍历文件夹下的所有文件和子文件夹
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除子文件夹
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'Folder {folder_path} does not exist.')

if __name__ == "__main__":
    get_result(product_name="水壺", product_feature="方便攜帶", gender="男性", age="18~25", job="", interest="運動體育", transparent_path=os.path.join(static_path, r"C:\Users\USER\Desktop\Develop\product-image-generator-backend\cv_integration\product_test\product_example_02_trans.png"), timestamp="test")