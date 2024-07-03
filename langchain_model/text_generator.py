import os
import json
import logging
import numpy as np
from dotenv import load_dotenv
from anthropic import Anthropic
import openai

# Load environment variables
load_dotenv()
claude_api_key = os.getenv("CLAUDE_API")
openai_api_key = os.getenv("CLAUDE_API")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure OpenAI API key
openai.api_key = openai_api_key

class TextGenerationError(Exception):
    "Custom exception for errors returned by Claude"
    def __init__(self, message):
        self.message = message

class AdGenerator:
    def __init__(self, embedding_model="text-embedding-ada-002"):
        self.embedding_model = embedding_model
        self.claude_client = Anthropic(api_key=claude_api_key)

    def create_prompt(self, product_name, product_describe, audience_type, length):
        """
        Create prompt for generating ad copy
        """
        if length == "短文":
            prompt = "用「繁體中文」生成一段吸引人的50-100字廣告文本。\n"
        elif length == "長文":
            prompt = "用「繁體中文」生成一段吸引人的200-300字廣告文本。\n"
        
        prompt += f"我們的產品名稱為 {product_name};"
        prompt += f"產品的特色是 {product_describe};"
        # prompt += f"產品圖片URL: {product_image_url};"
        prompt += f"希望此文案能讓 {audience_type} 產生興趣!\n"
        note = """
                1. 確認受眾是誰
                2. 吸睛的開頭
                3. 為什麼受眾想繼續看下去
                4. 凝練語句
                4.1 精簡
                4.2 點出重點且擺前面
                4.3 不要艱深
                5. 達成轉換
                文案的最後，當然是要達成促使受眾去做某件事。可能是下單、留下資料、領取優惠券，也可能是社群互動、閱讀更多文章等等。因此文案基本上都是以行動呼籲（Call to Action, CTA）結尾，附上產品連結、商家資訊等。
                """
        prompt += f"好的文案必須考慮以下:{note}"
        
        return prompt

    def generate_text(self, input_text, max_tokens=300):
        """
        Generate text using Claude Messages API
        """
        response = self.claude_client.messages.create(
            model="claude-1.3",
            system="You are a helpful advertising copywriter.",
            messages=[
                {"role": "user", "content": input_text}
            ],
            max_tokens=max_tokens
        )
        print("API Response:", response)

        # 提取生成的文本
        generated_text = ''.join([block.text for block in response.content])
        return generated_text

    def generate_ad_copy(self, product_name, product_describe, audience_type, length="短文"):
        """
        Generate ad copy
        """
        input_text = self.create_prompt(product_name, product_describe, audience_type, length)
        return self.generate_text(input_text)


if __name__ == "__main__":
    # Initialize Ad generator
    ad_generator = AdGenerator()

    # Define product information
    product_name = "水壺"
    product_describe = "高品質，可攜帶。"
    audience_type = "男性, 25-34, 軟體工程師, 運動"
    product_image_url = "http://example.com/image.jpg"

    # Generate short and long ad copy
    short_ad = ad_generator.generate_ad_copy(product_name, product_describe, audience_type, length="短文")
    long_ad = ad_generator.generate_ad_copy(product_name, product_describe, audience_type, length="長文")

    # Output generated ad copy
    print("Short Ad:", short_ad)
    print("Long Ad:", long_ad)
