import json
import logging
import os
import shutil
import boto3
from datetime import datetime
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv(".env")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
s3_bucket_name = os.getenv("S3_BUCKET_NAME")

# 初始化S3
s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# 讀取投放定向選項
def load_target_audiences():
    with open("target_audiences.json", "r", encoding="utf-8") as f:
        return json.load(f)

TARGET_AUDIENCES = load_target_audiences()

# 保存歷史紀錄到S3
def save_project_info_to_s3(project_info):
    filename = "static/history_setting.json"
    history = []

    try:
        # 從S3下載JSON
        try:
            s3_client.download_file(s3_bucket_name, filename, "/tmp/history_setting.json")
            with open("/tmp/history_setting.json", "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception as e:
            logging.info("No existing history_setting.json found in S3. Creating a new one.")

        history.append(project_info)

        # 將更新後的歷史紀錄上傳到S3
        with open("/tmp/history_setting.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

        s3_client.upload_file("/tmp/history_setting.json", s3_bucket_name, filename)
        logging.info(f"Saved project info to S3 at {filename}")
    except Exception as e:
        logging.error(f"Error saving project info to S3: {str(e)}")
        raise HTTPException(status_code=500, detail="Error saving project info to S3")

# 刪除只有upload的時間戳記文件夾
def cleanup_old_uploads():
    try:
        result = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix="static/")
        if 'Contents' in result:
            for obj in result['Contents']:
                if obj['Key'].endswith('/upload/'):
                    folder_prefix = obj['Key'].rsplit('/', 2)[0] + '/'
                    generated_prefix = folder_prefix + 'generated/'
                    generated_files = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix=generated_prefix)
                    if 'Contents' not in generated_files:
                        # 刪除沒有 generated 的 upload 文件夾
                        delete_objects = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix=folder_prefix)
                        if 'Contents' in delete_objects:
                            delete_keys = [{'Key': content['Key']} for content in delete_objects['Contents']]
                            s3_client.delete_objects(Bucket=s3_bucket_name, Delete={'Objects': delete_keys})
                            logging.info(f"Deleted old upload-only directory: {folder_prefix}")
    except Exception as e:
        logging.error(f"Error cleaning up old uploads: {str(e)}")
