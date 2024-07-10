import os
import logging
from dotenv import load_dotenv
import openai

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from configs import env_path

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

# Load environment variables
load_dotenv(env_path)
openai.api_key = os.getenv("OPENAI_KEY")

class TextGenerationError(Exception):
    """Custom exception for errors returned by OpenAI API"""
    def __init__(self, message):
        super().__init__(message)
        logger.error(f"TextGenerationError: {message}")

class AdGenerator:
    """Class to generate advertising copy using OpenAI's GPT models"""

    def __init__(self, model="gpt-3.5-turbo"):
        """
        Initialize the AdGenerator with the specified model.

        Args:
            model (str): The model to use for text generation.
        """
        self.model = model
        logger.info(f"AdGenerator initialized with model {model}")

    def create_prompt(self, product_name, product_describe, audience_type, length):
        """
        Create a prompt for generating ad copy.

        Args:
            product_name (str): The name of the product.
            product_describe (str): Description of the product.
            audience_type (str): The target audience for the ad.
            length (str): The desired length of the ad copy ("短文" or "長文").

        Returns:
            str: The generated prompt.
        """
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
        logger.info(f"Created prompt for product: {product_name}")
        return prompt + note

    def generate_text(self, input_text):
        """
        Generate text using OpenAI Chat API.

        Args:
            input_text (str): The input prompt for text generation.

        Returns:
            str: The generated text.

        Raises:
            TextGenerationError: If there is an error with the OpenAI API call.
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a helpful advertising copywriter."},
                          {"role": "user", "content": input_text}],
            )
            # Extract the generated text
            generated_text = response['choices'][0]['message']['content']
            logger.info("Text generation successful")
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise TextGenerationError(str(e))

    def generate_ad_copy(self, product_name, product_describe, audience_type, length="短文"):
        """
        Generate ad copy.

        Args:
            product_name (str): The name of the product.
            product_describe (str): Description of the product.
            audience_type (str): The target audience for the ad.
            length (str): The desired length of the ad copy ("短文" or "長文").

        Returns:
            str: The generated ad copy.
        """
        input_text = self.create_prompt(product_name, product_describe, audience_type, length)
        return self.generate_text(input_text)

if __name__ == "__main__":
    # Initialize the AdGenerator
    ad_generator = AdGenerator()

    # Define product information
    product_name = "水壺"
    product_describe = "高品質，可攜帶。"
    audience_type = "男性, 25-34, 軟體工程師, 運動"

    # Generate short and long ad copy
    try:
        short_ad = ad_generator.generate_ad_copy(product_name, product_describe, audience_type, length="短文")
        long_ad = ad_generator.generate_ad_copy(product_name, product_describe, audience_type, length="長文")

        # Output generated ad copy
        print("Short Ad:", short_ad)
        print("Long Ad:", long_ad)
        logger.info("Generated short and long ad copy successfully")
    except TextGenerationError as e:
        logger.error(f"Failed to generate ad copy: {str(e)}")
