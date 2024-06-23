import boto3
import logging
import uuid
import time
import os
from dotenv import load_dotenv

# 加載環境變量
load_dotenv(".env")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
s3_bucket_name = os.getenv("S3_BUCKET_NAME")

class BedrockClient:
    def __init__(self, region_name='ap-northeast-1'):
        '''
        初始化 BedrockClient，設定 AWS 客戶端
        '''
        self.client = boto3.client('bedrock', region_name=region_name,
                                   aws_access_key_id=aws_access_key_id, 
                                   aws_secret_access_key=aws_secret_access_key)

    def create_ad_images_job(self, job_name, custom_model_name, role_arn, base_model_identifier, product_image_s3_uri, target_keywords, output_data_s3_uri):
        '''
        創建廣告圖片生成任務
        
        :param job_name: 任務名稱
        :param custom_model_name: 自訂模型名稱
        :param role_arn: AWS 角色 ARN
        :param base_model_identifier: 基礎模型標識符
        :param product_image_s3_uri: 上傳圖片的 S3 URI
        :param target_keywords: 目標關鍵詞
        :param output_data_s3_uri: 輸出數據的 S3 URI
        :return: 任務的 ARN
        '''
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
            logging.info(f"創建廣告圖片生成任務，ARN: {response['jobArn']}")
            return response['jobArn']
        except Exception as e:
            logging.error(f"創建廣告圖片生成任務出錯: {str(e)}")
            return None

    def get_ad_images_job(self, job_arn):
        '''
        獲取廣告圖片生成任務的狀態與結果
        
        :param job_arn: 任務的 ARN
        :return: 任務的響應數據
        '''
        try:
            response = self.client.get_model_customization_job(jobIdentifier=job_arn)
            logging.info(f"獲取廣告圖片生成任務: {response}")
            return response
        except Exception as e:
            logging.error(f"獲取廣告圖片生成任務出錯: {str(e)}")
            return None

    def generate_ad_images(self, project_name, product_image_local_path, target_keywords, timestamp):
        '''
        生成廣告圖片
        
        :param project_name: 項目名稱
        :param product_image_local_path: 本地產品圖片路徑
        :param target_keywords: 目標關鍵詞
        :param timestamp: 上傳圖片的時間戳
        :return: 生成的圖片列表
        '''
        try:
            job_name = f"{project_name}_{timestamp}"
            custom_model_name = "custom_ad_model"
            role_arn = os.getenv("AWS_ROLE_ARN")
            base_model_identifier = "base-model"
            product_image_s3_uri = self.upload_to_s3(product_image_local_path, f"{timestamp}/upload/{os.path.basename(product_image_local_path)}")
            output_data_s3_uri = f"s3://{s3_bucket_name}/{timestamp}/generated/"

            job_arn = self.create_ad_images_job(
                job_name,
                custom_model_name,
                role_arn,
                base_model_identifier,
                product_image_s3_uri,
                target_keywords,
                output_data_s3_uri
            )

            if job_arn is None:
                raise Exception("創建廣告圖片生成任務失敗")

            # 輕度輪詢任務狀態（簡化示例，實際應用中應使用更可靠的方法）
            job_status = None
            while job_status not in ["COMPLETED", "FAILED"]:
                job_response = self.get_ad_images_job(job_arn)
                job_status = job_response['status']
                logging.info(f"任務狀態: {job_status}")
                if job_status == "COMPLETED":
                    generated_images = self.list_generated_images(output_data_s3_uri)
                    return generated_images
                elif job_status == "FAILED":
                    raise Exception("廣告圖片生成任務失敗")
                time.sleep(10)  # 等待一段時間後再次輪詢

        except Exception as e:
            logging.error(f"生成廣告圖片出錯: {str(e)}")
            return []

    def upload_to_s3(self, local_path, s3_key):
        '''
        上傳文件到 S3
        
        :param local_path: 本地文件路徑
        :param s3_key: S3 key
        :return: 上傳的 S3 URI
        '''
        try:
            s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
            s3_client.upload_file(local_path, s3_bucket_name, s3_key)
            s3_uri = f"s3://{s3_bucket_name}/{s3_key}"
            logging.info(f"上傳 {local_path} 到 {s3_uri}")
            return s3_uri
        except Exception as e:
            logging.error(f"上傳到 S3 出錯: {str(e)}")
            return None

    def list_generated_images(self, s3_uri_prefix):
        '''
        列出生成的圖片
        
        :param s3_uri_prefix: S3 URI 前綴
        :return: 生成的圖片列表
        '''
        try:
            s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
            prefix = s3_uri_prefix.replace(f"s3://{s3_bucket_name}/", "")
            response = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix=prefix)
            if 'Contents' not in response:
                return []
            generated_images = [f"s3://{s3_bucket_name}/{item['Key']}" for item in response['Contents'] if item['Key'].endswith('.jpg')]
            return generated_images
        except Exception as e:
            logging.error(f"列出生成的圖片出錯: {str(e)}")
            return []
