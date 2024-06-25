import boto3
import logging
import json
import os
import threading
from dotenv import load_dotenv

# 加載變數
load_dotenv(r"C:\Users\USER\Desktop\Develop\product-image-generator-backend\.env")
aws_access_key_id = os.getenv("AWS_KEY")
aws_secret_access_key = os.getenv("AWS_SECRET_KEY")
region_name = os.getenv("AWS_REGION", "ap-northeast-1")

class TextGenerationError(Exception):
    "Custom exception for errors returned by Amazon Titan Text models"
    def __init__(self, message):
        self.message = message

class TextGenerator:
    def __init__(self, model_id="amazon.titan-text-express-v1"):
        self.model_id = model_id
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def generate_text(self, input_text, max_token_count=8192, temperature=0, top_p=1.0):
        result = {"outputText": None, "error": None}
        def target():
            try:
                body = json.dumps({
                    "inputText": input_text,
                    "textGenerationConfig": {
                        "maxTokenCount": max_token_count,
                        "stopSequences": [],
                        "temperature": temperature,
                        "topP": top_p
                    }
                })
                response = self.client.invoke_model(
                    body=body,
                    modelId=self.model_id,
                    accept="application/json",
                    contentType="application/json"
                )

                result_data = json.loads(response['body'].read().decode('utf-8'))
                if "error" in result_data:
                    result["error"] = f"Text generation error: {result_data['error']}"
                else:
                    result["outputText"] = result_data.get('results')[0].get('outputText', 'Text generation failed')
            except Exception as e:
                result["error"] = f"Error generating text: {e}"

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=30)  # 30秒超時

        if thread.is_alive():
            self.logger.error("Text generation timed out.")
            thread.join()  # 等待線程結束
            raise TextGenerationError("Text generation timed out.")
        elif result["error"]:
            self.logger.error(result["error"])
            raise TextGenerationError(result["error"])

        return result["outputText"]

    def generate_ad_copy(self, product_name, product_describe, audience_type, length="short"):
        input_text = self.create_prompt(product_name, product_describe, audience_type, length)
        print(input_text)
        max_token_count = 256 if length == "short" else 8192
        return self.generate_text(input_text, max_token_count)

    def create_prompt(self, product_name, product_describe, audience_type, length):
        prompt = f"產品名稱: {product_name}.\n"
        prompt += f"產品特性: {product_describe}.\n\n"
        prompt += f"用繁體中文生成一則 {length} 廣告文本基於以上資訊及，以下為我們的受眾:\n"
        prompt += f"{audience_type}."
        return prompt

if __name__ == "__main__":
    text_generator = TextGenerator()
    product_name = "水壺"
    product_describe = "高品質，可攜帶。"
    audience_type = "男性,25-34,學生,旅行,閱讀,18-24,運動"
    
    try:
        short_ad = text_generator.generate_ad_copy(product_name, product_describe, audience_type, length="short")
        long_ad = text_generator.generate_ad_copy(product_name, product_describe, audience_type, length="long")
        print("Short Ad:", short_ad)
        print("Long Ad:", long_ad)
    except TextGenerationError as e:
        print(e.message)
