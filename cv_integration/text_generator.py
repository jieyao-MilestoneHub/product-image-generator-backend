import os
import logging
from dotenv import load_dotenv
import openai

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from configs import env_path

# Load environment variables
load_dotenv(env_path)
openai_api_key = os.getenv("OPENAI_API")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure OpenAI API key
openai.api_key = openai_api_key

class TextGenerationError(Exception):
    """Custom exception for errors returned by OpenAI API"""
    def __init__(self, message):
        super().__init__(message)

class AdGenerator:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    def create_prompt(self, product_name, product_describe, audience_type, length):
        """Create prompt for generating ad copy"""
        prompt_length = "50-100字" if length == "短文" else "200-300字"
        prompt = f"用「繁體中文」生成一段吸引人的{prompt_length}廣告文本。\n"
        prompt += f"產品名稱為 {product_name}; 產品的特色是 {product_describe}; 希望此文案能讓 {audience_type} 產生興趣!\n"
        note = """
        好的文案必須考慮以下:
        1. 確認受眾是誰
        2. 吸睛的開頭
        3. 為什麼受眾想繼續看下去
        4. 凝練語句 (精簡、點出重點且擺前面、不要艱深)
        5. 達成轉換 (以行動呼籲結尾，附上產品連結、商家資訊等)
        """
        return prompt + note

    def generate_text(self, input_text):
        """Generate text using OpenAI Chat API"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a helpful advertising copywriter."},
                        {"role": "user", "content": input_text}],
            )
            # 提取生成的文本
            generated_text = response['choices'][0]['message']['content']
            return generated_text.strip()
        except Exception as e:
            raise TextGenerationError(str(e))

    def generate_ad_copy(self, product_name, product_describe, audience_type, length="短文"):
        """Generate ad copy"""
        input_text = self.create_prompt(product_name, product_describe, audience_type, length)
        return self.generate_text(input_text)

if __name__ == "__main__":
    ad_generator = AdGenerator()

    # Define product information
    product_name = "水壺"
    product_describe = "高品質，可攜帶。"
    audience_type = "男性, 25-34, 軟體工程師, 運動"

    # Generate short and long ad copy
    short_ad = ad_generator.generate_ad_copy(product_name, product_describe, audience_type, length="短文")
    long_ad = ad_generator.generate_ad_copy(product_name, product_describe, audience_type, length="長文")

    # Output generated ad copy
    print("Short Ad:", short_ad)
    print("Long Ad:", long_ad)
