import os
import logging
import shutil
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import json
import pandas as pd
import io
from cv_integration.text_generator import AdGenerator
from cv_integration.image_generator import get_product, get_result
from configs import static_path

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('app.log', mode='a', encoding='utf-8')
file_handler.setFormatter(log_formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)

app_logger = logging.getLogger()
app_logger.setLevel(logging.INFO)
app_logger.addHandler(file_handler)
app_logger.addHandler(stream_handler)

# Ensure that imported modules use the same logging configuration
logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])

app = FastAPI()

# Allowed frontend URLs
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

with open('mock_dynamo_db.json', 'r') as json_file:
    data = json.load(json_file)

# Convert the read data into a dictionary
MOCK_DYNAMO_DB = {k: v for k, v in data.items()}

GENDER_TAGS = {
    "1001": "男性",
    "1002": "女性"
}

AGE_TAGS = {
    "2001": "18-24",
    "2002": "25-34",
    "2003": "35-44",
    "2004": "45-54",
    "2005": "55+"
}

# OCCUPATION_TAGS = {
#     "3001": "學生",
#     "3002": "軟體工程師",
#     "3003": "醫生",
#     "3004": "教師",
#     "3005": "其他"
# }

INTEREST_TAGS = {
    "4001": "運動體育",
    "4002": "寵物生活",
    "4003": "嬰幼保健",
    "4004": "動漫電競",
    "4005": "戶外旅遊"
}

class AdGenerateRequest(BaseModel):
    product_name: str
    product_describe: str
    target_audience: str
    product_image_url: str

class ImageGenerateRequest(BaseModel):
    product_name: str
    product_describe: str
    target_audience: str
    product_image_filename: str
    timestamp: str

# Create directory to store images
STATIC_DIR = static_path
os.makedirs(STATIC_DIR, exist_ok=True)

# Load targeting options
def load_target_audiences() -> Dict[str, str]:
    with open("target_audiences.json", "r", encoding="utf-8") as f:
        return json.load(f)

TARGET_AUDIENCES = load_target_audiences()

# Save project information to JSON
def save_project_info(project_info: Dict):
    filename = os.path.join(STATIC_DIR, "history_setting.json")
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

# Delete folders that only have upload timestamp
def cleanup_old_uploads():
    for folder_name in os.listdir(STATIC_DIR):
        folder_path = os.path.join(STATIC_DIR, folder_name)
        if os.path.isdir(folder_path):
            upload_path = os.path.join(folder_path, "upload")
            generated_path = os.path.join(folder_path, "generated")
            if os.path.exists(upload_path) and not os.path.exists(generated_path):
                shutil.rmtree(folder_path)
                logging.info(f"Deleted old upload-only directory: {folder_path}")

# Upload CSV and return label analysis results
@app.post("/api/upload-audience")
async def upload_audience(file: UploadFile = File(...)):
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Build corresponding structure
        gender_labels = []
        age_labels = []
        # occupation_labels = []
        interest_labels = []
        for index, row in df.iterrows():
            uid = row['uid']
            if uid in MOCK_DYNAMO_DB:
                labels = MOCK_DYNAMO_DB[uid]
                for label in labels:
                    for tag in label.split(','):
                        if tag in GENDER_TAGS:
                            gender_labels.append(tag)
                        elif tag in AGE_TAGS:
                            age_labels.append(tag)
                        # elif tag in OCCUPATION_TAGS:
                        #     occupation_labels.append(tag)
                        elif tag in INTEREST_TAGS:
                            interest_labels.append(tag)
        
        # Count occurrences of each label
        def count_labels(label_list, tags_dict):
            label_counts = {name: 0 for name in tags_dict.values()}
            for tag in label_list:
                label_counts[tags_dict[tag]] += 1
            return {"labels": list(label_counts.keys()), "values": list(label_counts.values())}
        
        gender_data = count_labels(gender_labels, GENDER_TAGS)
        age_data = count_labels(age_labels, AGE_TAGS)
        # occupation_data = count_labels(occupation_labels, OCCUPATION_TAGS)
        interest_data = count_labels(interest_labels, INTEREST_TAGS)

        return {
            "gender_data": gender_data,
            "age_data": age_data,
            # "occupation_data": occupation_data,
            "interest_data": interest_data
        }
    except Exception as e:
        logging.error(f"Error processing audience upload: {str(e)}")
        return {"error": str(e)}

@app.get("/api/target-audiences")
async def get_target_audiences():
    return JSONResponse(content=TARGET_AUDIENCES)

@app.post("/api/upload-image")
async def upload_image(product_image: UploadFile = File(...)):
    try:
        # Clean up unused directories
        cleanup_old_uploads()

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        upload_dir = os.path.join(STATIC_DIR, timestamp, "upload")
        os.makedirs(upload_dir, exist_ok=True)
        upload_path = os.path.join(upload_dir, product_image.filename)

        # Save uploaded image
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(product_image.file, buffer)

        logging.info(f"Uploaded image: {timestamp}/upload/{product_image.filename}")

        # Make background transparent
        trans_path = get_product(upload_path, STATIC_DIR, timestamp)

        logging.info(f"Preprocess image: {timestamp}/upload/{product_image.filename}")
        return JSONResponse(content={"filename": f"{timestamp}/upload/{product_image.filename}", "timestamp": timestamp, "trans_path": trans_path})
    except Exception as e:
        logging.error(f"Error uploading image: {str(e)}")
        return JSONResponse(content={"error": "Error uploading image"}, status_code=500)

@app.post("/api/generate-product")
async def generate_product(
    product_name: str = Form(...),
    product_describe: str = Form(...),
    target_audience: str = Form(...),
    product_image_filename: str = Form(...),
    timestamp: str = Form(...)
):
    try:
        trans_path_prefix = os.path.join(STATIC_DIR, "product_transparent", timestamp)
        img_product = os.path.basename(product_image_filename).split(".")[0]
        transparent_path = os.path.join(trans_path_prefix, f"{img_product}_transparent.png")

        if not os.path.exists(transparent_path):
            logging.error(f"Transparent file {transparent_path} does not exist")
            raise HTTPException(status_code=400, detail="Transparent image is still processing, please wait")

        # Generate images
        generated_path = os.path.join(STATIC_DIR, timestamp, "generated")
        background_path = os.path.join(STATIC_DIR, timestamp, "background")
        os.makedirs(generated_path, exist_ok=True)
        logging.info("Generating images")
        audience = target_audience.split(",")
        gender = audience[0]
        age = audience[1]
        job = "no data"
        interest = audience[2]
        generated_images = get_result(product_name, product_describe, gender, age, job, interest, transparent_path, generated_path=generated_path, background_path=background_path)

        if not generated_images:
            return JSONResponse(content={"error": "get_result() got the problem"}, status_code=404)

        # Generate text
        logging.info("Generating text")
        ad_generator = AdGenerator()
        short_ad = ad_generator.generate_ad_copy(product_name, product_describe, target_audience, length="短文")
        long_ad = ad_generator.generate_ad_copy(product_name, product_describe, target_audience, length="長文")

        logging.info(f"Received product_name: {product_name}")
        logging.info(f"Received product_describe: {product_describe}")
        logging.info(f"Received target_audience: {target_audience}")
        logging.info(f"Received product_image_filename: {product_image_filename}")
        logging.info(f"Received timestamp: {timestamp}")

        project_info = {
            "write_date": timestamp,
            "product_describe": product_describe,
            "product_name": product_name,
            "target_audience": target_audience,
            "product_image_filename": product_image_filename,
            "generated_images": generated_images,
            "short_ad": short_ad,
            "long_ad": long_ad
        }

        save_project_info(project_info)

        return JSONResponse(content=project_info)

    except HTTPException as he:
        logging.error(f"HTTP error: {str(he)}")
        return JSONResponse(content={"error": he.detail}, status_code=he.status_code)
    except Exception as e:
        logging.error(f"Error generating project: {str(e)}")
        return JSONResponse(content={"error": "Error generating project"}, status_code=500)

@app.get("/api/history")
async def get_history():
    filename = os.path.join(STATIC_DIR, "history_setting.json")
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            history = json.load(f)
            return JSONResponse(content=history)
    else:
        return JSONResponse(content=[], status_code=200)
    
# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
