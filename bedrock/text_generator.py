import boto3
import logging
import json
import os
from dotenv import load_dotenv

# 載入環境變數
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
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def generate_text(self, input_text, max_token_count=8192, temperature=0, top_p=1.0):
        result = {"outputText": None, "error": None}
        try:
            self.logger.info("Generating text with input: %s", input_text)
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
            self.logger.info("Text generation result: %s", result["outputText"])
        except Exception as e:
            result["error"] = f"Error generating text: {e}"
            self.logger.error("Error generating text: %s", e)

        return result["outputText"]

    def categorize_tags(self, tags):
        input_text = f"請將以下標籤分類為年齡標籤、性別標籤、興趣標籤和職業標籤: {tags}"
        self.logger.info("Categorizing tags with input: %s", input_text)
        response_text = self.generate_text(input_text, max_token_count=512)
        self.logger.info("Categorized tags result: %s", response_text)
        return response_text

    def generate_character_features(self, audience_type):
        categorized_tags = self.categorize_tags(audience_type)
        character_features = self.extract_character_features(categorized_tags)
        return character_features

    def extract_character_features(self, categorized_tags):
        lines = categorized_tags.split('\n')
        age_tag = next((line for line in lines if "年齡標籤" in line), "")
        gender_tag = next((line for line in lines if "性別標籤" in line), "")
        interest_tag = next((line for line in lines if "興趣標籤" in line), "")
        occupation_tag = next((line for line in lines if "職業標籤" in line), "")
        character_features = f"{age_tag} {gender_tag} {occupation_tag} {interest_tag}"
        return character_features

    def generate_ad_copy(self, product_name, product_describe, audience_type, length="短文"):
        character_features = self.generate_character_features(audience_type)
        print(character_features)
        input_text = self.create_prompt(product_name, product_describe, character_features, length)
        # max_token_count = 100 if length == "短文" else 300
        return self.generate_text(input_text)

    def create_prompt(self, product_name, product_describe, character_features, length):
        prompt = f"產品名稱: {product_name}。\n"
        prompt += f"產品特性: {product_describe}。\n"
        prompt += f"受眾: {character_features}。\n\n"
        if length == "短文":
            prompt += f"用「繁體中文」生成一句吸引人的「50-100」字的廣告文本，吸引該客群來採購，不能換行與列點!\n\n"
        elif length == "長文":
            prompt += f"用「繁體中文」生成一句吸引人的「200-300」字的廣告文本，吸引該客群來採購，不能換行與列點!\n\n"            
        return prompt

    def prompt_top_combinations(self, audience_types, n=5):
        audience_types_str = ", ".join(audience_types)
        input_text = f"請基於以下的受眾標籤生成最有可能的五種受眾特徵組合，每個組合應包含年齡、性別、職業和興趣標籤: {audience_types_str}"
        self.logger.info("Generating top combinations with input: %s", input_text)
        response_text = self.generate_text(input_text)
        combinations = response_text.split('\n')
        self.logger.info("Generated combinations: %s", combinations)
        return combinations

if __name__ == "__main__":
    text_generator = TextGenerator()
    product_name = "水壺"
    product_describe = "高品質，可攜帶。"
    
    options = [
        "男性", "女性", "18-24", "25-34", "35-44", "45-54", "55+",
        "學生", "軟體工程師", "醫生", "教師", "其他",
        "音樂", "運動", "旅行", "閱讀", "遊戲"
    ]
    
    selected_combinations = text_generator.prompt_top_combinations(options, n=5)
    
    try:
        for audience_type in selected_combinations:
            print(f"Selected combination: {audience_type}")
            short_ad = text_generator.generate_ad_copy(product_name, product_describe, audience_type, length="短文")
            print("Short Ad:", short_ad)
    except TextGenerationError as e:
        print(e.message)
