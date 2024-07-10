# User Guide

This project aims to achieve precise targeting through AIGC and needs to be used in conjunction with [`product-image-generator-frontend`](https://github.com/jieyao-MilestoneHub/product-image-generator-frontend).

## Tech Stack

- **Backend Framework**: FastAPI
- **Deep Learning**: PyTorch
- **Image Processing**: OpenCV, PIL
- **Models**: Mask R-CNN, DeepFill, Segment Anything
- **Cloud Services**: AWS S3, BedrockClient

## Model Weights

- **DeepFill**: Model weights can be downloaded from [generative_inpainting](https://github.com/JiahuiYu/generative_inpainting).
- **Mask R-CNN**: Model weights can be downloaded from [Mask_RCNN](https://github.com/matterport/Mask_RCNN).
- **Segment Anything**: Model weights can be downloaded from [segment-anything](https://github.com/facebookresearch/segment-anything).

After downloading the above weights, place the files as referenced in `~/configs.py`.

## Installation and Running

1. Clone the repository:
    ```bash
    git clone <repository-url>
    ```

2. Navigate to the project directory:
    ```bash
    cd product-image-generator-backend
    ```

3. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:
    - Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    - Unix or MacOS:
        ```bash
        source venv/bin/activate
        ```

5. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

6. Create a `.env` file and add your AWS keys:
    ```plaintext
    AWS_ACCESS_KEY_ID=your_access_key_id
    AWS_SECRET_ACCESS_KEY=your_secret_access_key
    S3_BUCKET_NAME=your_s3_bucket_name
    AWS_ROLE_ARN=your_aws_role_arn
    OPENAI_KEY=your_openai_key
    ```

7. Run the application:
    ```bash
    uvicorn main:app --reload
    ```

8. You can now use the frontend page seamlessly.

## History Record Saving Logic

1. **Image Upload**:
    - Each time an image is uploaded, the system saves the image to the `static/{timestamp}/upload/` directory in S3, where `{timestamp}` is the upload timestamp of the image.
    - Before uploading an image, the system cleans up all old timestamp directories that only have the `upload` subdirectory and no `generated` subdirectory.

2. **Generate Images**:
    - When generating images, the system downloads the uploaded images from S3, invokes the BedrockClient to generate the images, and saves the generated images to the `static/{timestamp}/generated/` directory in S3.
    - If the image generation fails, the system deletes the uploaded images.

3. **History Record Saving**:
    - Each time an image is successfully generated, the system saves the generated image information and project information to the `static/history_setting.json` file and uploads it to S3.
    - The history record includes the project name, targeting details, uploaded image filename, and a list of generated images.
    - When calling the history record API, the system downloads the `history_setting.json` file from S3 and returns the history record information.
