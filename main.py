from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
import shutil
import os
import logging
import json
from datetime import datetime
# from bedrock.bedrock_client import BedrockClient

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

# 創建儲存圖片的目錄
UPLOAD_DIR = "uploads"
GENERATED_DIR = "generated"
PROJECTS_DIR = "history_projects"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)
os.makedirs(PROJECTS_DIR, exist_ok=True)

# 讀取投放定向選項
def load_target_audiences() -> Dict[str, str]:
    with open("target_audiences.json", "r", encoding="utf-8") as f:
        return json.load(f)

TARGET_AUDIENCES = load_target_audiences()

# 保存歷史紀錄到JSON
def save_project_info(project_info: Dict):
    filename = f"{PROJECTS_DIR}/history_setting.json"
    try:
        if os.path.exists(filename):
            with open(filename, "r+", encoding="utf-8") as f:
                history = json.load(f)
                history.append(project_info)
                f.seek(0)
                json.dump(history, f, ensure_ascii=False, indent=4)
        else:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump([project_info], f, ensure_ascii=False, indent=4)
        logging.info(f"Saved project info to {filename}")
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error saving project info: JSON decode error")
    except PermissionError as e:
        logging.error(f"Permission error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error saving project info: Permission error")
    except Exception as e:
        logging.error(f"Unexpected error saving project info: {str(e)}")
        raise HTTPException(status_code=500, detail="Error saving project info: Unexpected error")

@app.get("/api/target-audiences")
async def get_target_audiences():
    return JSONResponse(content=TARGET_AUDIENCES)

@app.post("/api/upload-image")
async def upload_image(product_image: UploadFile = File(...)):
    try:
        # 保存上傳的圖片
        upload_path = os.path.join(UPLOAD_DIR, product_image.filename)
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(product_image.file, buffer)

        logging.info(f"Uploaded image: {product_image.filename}")
        return JSONResponse(content={"filename": product_image.filename})
    except Exception as e:
        logging.error(f"Error uploading image: {str(e)}")
        return JSONResponse(content={"error": "Error uploading image"}, status_code=500)

@app.post("/api/generate-images")
async def generate_images(
    project_name: str = Form(...),
    target_audience: str = Form(...),
    product_image_filename: str = Form(...)
):
    try:
        logging.info(f"Received project_name: {project_name}")
        logging.info(f"Received target_audience: {target_audience}")
        logging.info(f"Received product_image_filename: {product_image_filename}")

        upload_path = os.path.join(UPLOAD_DIR, product_image_filename)
        if not os.path.exists(upload_path):
            logging.error(f"Uploaded file {upload_path} does not exist")
            raise HTTPException(status_code=400, detail="Uploaded file does not exist")

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Simulate generating images and saving them
        generated_images = [f"{timestamp}_generated_{i}.jpg" for i in range(1, 4)]
        for image_name in generated_images:
            generated_path = os.path.join(GENERATED_DIR, image_name)
            shutil.copyfile(upload_path, generated_path)

        logging.info(f"Generated images for project {project_name}, target audience: {target_audience}")

        # Save all historical records to a dictionary
        project_info = {
            "write_date": timestamp,
            "project_name": project_name,
            "target_audience": target_audience,
            "product_image_filename": product_image_filename,
            "generated_images": generated_images
        }

        save_project_info(project_info)

        return JSONResponse(content=project_info)
    except HTTPException as he:
        logging.error(f"HTTP error: {str(he)}")
        return JSONResponse(content={"error": he.detail}, status_code=he.status_code)
    except Exception as e:
        logging.error(f"Error generating images: {str(e)}")
        return JSONResponse(content={"error": "Error generating images"}, status_code=500)

# 真正使用 Bedrock
# @app.post("/api/generate-images")
# async def generate_images(
#     project_name: str = Form(...),
#     target_audience: str = Form(...),
#     product_image_filename: str = Form(...)
# ):
#     try:
#         logging.info(f"Received project_name: {project_name}")
#         logging.info(f"Received target_audience: {target_audience}")
#         logging.info(f"Received product_image_filename: {product_image_filename}")

#         timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

#         # 配置AWS Bedrock的必要参数
#         bedrock_client = BedrockClient()
#         job_name = f"{project_name}_{timestamp}"
#         custom_model_name = f"{project_name}_model"
#         role_arn = 'arn:aws:iam::YOUR_AWS_ACCOUNT_ID:role/YOUR_ROLE_NAME'
#         base_model_identifier = 'base-model-id'  # 基礎模型的標示符
#         product_image_s3_uri = f"s3://your-bucket/{product_image_filename}"
#         target_keywords = target_audience
#         output_data_s3_uri = f"s3://your-bucket/output/{timestamp}"

#         job_arn = bedrock_client.create_ad_images_job(
#             job_name=job_name,
#             custom_model_name=custom_model_name,
#             role_arn=role_arn,
#             base_model_identifier=base_model_identifier,
#             product_image_s3_uri=product_image_s3_uri,
#             target_keywords=target_keywords,
#             output_data_s3_uri=output_data_s3_uri
#         )

#         if job_arn:
#             job_result = bedrock_client.get_ad_images_job(job_arn)
#             generated_images = [f"{timestamp}_{i}.jpg" for i in range(len(job_result['GeneratedImages']))]
#             for i, image_data in enumerate(job_result['GeneratedImages']):
#                 with open(os.path.join(GENERATED_DIR, generated_images[i]), 'wb') as output_file:
#                     output_file.write(image_data['Bytes'])

#             logging.info(f"Generated images for project {project_name}, target audience: {target_audience}")
        
#             # 保存所有文字信息到字典
#             project_info = {
#                 "write_date": timestamp,
#                 "project_name": project_name,
#                 "target_audience": target_audience,
#                 "product_image_filename": product_image_filename,
#                 "generated_images": generated_images
#             }
#             save_project_info(project_info)

#             return JSONResponse(content=project_info)
#         else:
#             return JSONResponse(content={"error": "Error creating ad images job"}, status_code=500)
#     except Exception as e:
#         logging.error(f"Error generating images: {str(e)}")
#         return JSONResponse(content={"error": "Error generating images"}, status_code=500)


# 提供靜態文件的服務
app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
