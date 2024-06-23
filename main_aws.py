import logging
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from utils import cleanup_old_uploads, load_target_audiences, save_project_info_to_s3
from bedrock_client import BedrockClient
import boto3
from dotenv import load_dotenv
import os

# 加載環境變量
load_dotenv(".env")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
s3_bucket_name = os.getenv("S3_BUCKET_NAME")

# 配置日誌
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = FastAPI()

# 允許的前端URL
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 BedrockClient 和 S3 客戶端
bedrock_client = BedrockClient()
s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# 讀取投放定向選項
TARGET_AUDIENCES = load_target_audiences()

@app.get("/api/target-audiences")
async def get_target_audiences():
    """
    獲取投放定向選項
    """
    return JSONResponse(content=TARGET_AUDIENCES)

@app.post("/api/upload-image")
async def upload_image(product_image: UploadFile = File(...)):
    """
    上傳圖片並保存到 S3
    """
    try:
        # 清理舊的 upload-only 文件夾
        cleanup_old_uploads()

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        upload_key = f"static/{timestamp}/upload/{product_image.filename}"

        # 上傳圖片到 S3
        s3_client.upload_fileobj(product_image.file, s3_bucket_name, upload_key)

        logging.info(f"Uploaded image: {product_image.filename}")
        return JSONResponse(content={"filename": upload_key, "timestamp": timestamp})
    except Exception as e:
        logging.error(f"Error uploading image: {str(e)}")
        return JSONResponse(content={"error": "Error uploading image"}, status_code=500)

@app.post("/api/generate-images")
async def generate_images(
    project_name: str = Form(...),
    target_audience: str = Form(...),
    product_image_filename: str = Form(...),
    timestamp: str = Form(...)
):
    """
    生成圖片並保存到 S3
    """
    try:
        logging.info(f"Received project_name: {project_name}")
        logging.info(f"Received target_audience: {target_audience}")
        logging.info(f"Received product_image_filename: {product_image_filename}")
        logging.info(f"Received timestamp: {timestamp}")

        upload_key = product_image_filename
        upload_path = f"/tmp/{os.path.basename(upload_key)}"

        # 下載上傳的圖片
        s3_client.download_file(s3_bucket_name, upload_key, upload_path)

        target_keywords = target_audience.split(',')
        generated_images = bedrock_client.generate_ad_images(project_name, upload_path, target_keywords, timestamp)

        if not generated_images:
            # 如果生成失敗，刪除上傳的圖片
            s3_client.delete_object(Bucket=s3_bucket_name, Key=upload_key)
            logging.error(f"Failed to generate images for project {project_name}")
            raise HTTPException(status_code=500, detail="Failed to generate images")

        logging.info(f"Generated images for project {project_name}, target audience: {target_audience}")

        # 保存所有歷史紀錄到字典
        project_info = {
            "write_date": timestamp,
            "project_name": project_name,
            "target_audience": target_audience,
            "product_image_filename": upload_key,
            "generated_images": generated_images
        }

        save_project_info_to_s3(project_info)

        return JSONResponse(content=project_info)
    except HTTPException as he:
        logging.error(f"HTTP error: {str(he)}")
        return JSONResponse(content={"error": he.detail}, status_code=he.status_code)
    except Exception as e:
        logging.error(f"Error generating images: {str(e)}")
        # 刪除上傳的圖片
        if os.path.exists(upload_path):
            s3_client.delete_object(Bucket=s3_bucket_name, Key=upload_key)
        return JSONResponse(content={"error": "Error generating images"}, status_code=500)

@app.get("/api/history")
async def get_history():
    """
    獲取歷史紀錄
    """
    try:
        filename = "static/history_setting.json"
        # 從 S3 下載 history_setting.json 文件
        s3_client.download_file(s3_bucket_name, filename, "/tmp/history_setting.json")
        with open("/tmp/history_setting.json", "r", encoding="utf-8") as f:
            history = json.load(f)
            return JSONResponse(content=history)
    except Exception as e:
        logging.error(f"Error fetching history: {str(e)}")
        return JSONResponse(content=[], status_code=500)

# 提供靜態文件的服務
app.mount("/static", StaticFiles(directory="static"), name="static")
