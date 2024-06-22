import boto3
import logging
import uuid
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(".env")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

class BedrockClient:
    def __init__(self, region_name='ap-northeast-1'):
        '''
        create_ad_images_job: 創建廣告圖片素材。
        get_ad_images_job: 取得生成狀態與結果。
        '''
        self.client = boto3.client('bedrock', region_name=region_name,
                                   aws_access_key_id=aws_access_key_id, 
                                   aws_secret_access_key=aws_secret_access_key)

    def create_ad_images_job(self, job_name, custom_model_name, role_arn, base_model_identifier, product_image_s3_uri, target_keywords, output_data_s3_uri):
        try:
            response = self.client.create_model_customization_job(
                jobName=job_name,
                customModelName=custom_model_name,
                roleArn=role_arn,
                clientRequestToken=str(uuid.uuid4()),
                baseModelIdentifier=base_model_identifier,
                customizationType='FINE_TUNING',
                trainingDataConfig={'s3Uri': product_image_s3_uri},
                outputDataConfig={'s3Uri': output_data_s3_uri},
                hyperParameters={'target_keywords': target_keywords}
            )
            logging.info(f"Created ad images job with ARN: {response['jobArn']}")
            return response['jobArn']
        except Exception as e:
            logging.error(f"Error creating ad images job: {str(e)}")
            return None

    def get_ad_images_job(self, job_arn):
        try:
            response = self.client.get_model_customization_job(jobIdentifier=job_arn)
            logging.info(f"Retrieved ad images job: {response}")
            return response
        except Exception as e:
            logging.error(f"Error getting ad images job: {str(e)}")
            return None
