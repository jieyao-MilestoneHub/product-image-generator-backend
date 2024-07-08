import openai
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
import sys
from configs import import_path, env_path
sys.path.append(import_path)

# 加載環境變量
load_dotenv(env_path)
openai.api_key = os.getenv("OPENAI_KEY")

# 使用 GPT 模型進行文本翻譯
def translate_text_gpt(text, model="gpt-3.5-turbo"):
    if not openai.api_key:
        raise ValueError("OpenAI API key is not set. Please set the API_KEY environment variable.")
    
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