import random
import os
import base64
import io
import json
import logging
import time
from PIL import Image
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
from enum import Enum, unique

# Load environment variables
load_dotenv()

# AWS Credentials
AWS_KEY_ID = os.getenv("AWS_KEY_ID")
AWS_KEY_SECRET = os.getenv("AWS_KEY_SECRET")

# Constants
MODEL_ID = "stability.stable-diffusion-xl-v1"
GENERATED_IMAGES = "./content/generated_images"

class ImageError(Exception):
    """Custom exception for errors returned by SDXL 1.0."""
    def __init__(self, message):
        self.message = message

def setup_logger():
    """Set up the logger."""
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = setup_logger()

class ImageToImageRequest:
    """Class for handling image to image request parameters."""
    def __init__(self, image_width, image_height, positive_prompt, negative_prompt,
                 init_image_mode="IMAGE_STRENGTH", image_strength=0.5, cfg_scale=7,
                 clip_guidance_preset="SLOWER", sampler="K_DPMPP_2M", samples=1,
                 seed=1, steps=30, style_preset="photographic", extras=None):
        self.image_width = image_width
        self.image_height = image_height
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.init_image_mode = init_image_mode
        self.image_strength = image_strength
        self.cfg_scale = cfg_scale
        self.clip_guidance_preset = clip_guidance_preset
        self.sampler = sampler
        self.samples = samples
        self.seed = seed
        self.steps = steps
        self.style_preset = style_preset
        self.extras = extras

@unique
class StylesPresets(Enum):
    """Enumerator for SDXL style presets."""
    THREE_D_MODEL = "3d-model"
    ANALOG_FILM = "analog-film"
    ANIME = "anime"
    CINEMATIC = "cinematic"
    COMIC_BOOK = "comic-book"
    DIGITAL_ART = "digital-art"
    ENHANCE = "enhance"
    FANTASY_ART = "fantasy-art"
    ISOMETRIC = "isometric"
    LINE_ART = "line-art"
    LOW_POLY = "low-poly"
    MODELING_COMPOUND = "modeling-compound"
    NEON_PUNK = "neon-punk"
    ORIGAMI = "origami"
    PHOTOGRAPHIC = "photographic"
    PIXEL_ART = "pixel-art"
    TILE_TEXTURE = "tile-texture"

def get_bedrock_client():
    """Initialize and return the Bedrock client."""
    return boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        aws_access_key_id=AWS_KEY_ID,
        aws_secret_access_key=AWS_KEY_SECRET
    )

def generate_image_from_image(model_id, body):
    """Generate an image using SDXL 1.0 on demand."""
    logger.info("Generating image with SDXL model %s", model_id)
    bedrock = get_bedrock_client()
    response = bedrock.invoke_model(
        body=body, modelId=model_id, accept="application/json", contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    logger.info(f"Bedrock result: {response_body['result']}")
    artifact = response_body.get("artifacts")[0]
    base64_image = artifact.get("base64")
    image_bytes = base64.b64decode(base64_image.encode("ascii"))
    finish_reason = artifact.get("finishReason")
    if finish_reason in ["ERROR", "CONTENT_FILTERED"]:
        raise ImageError(f"Image generation error. Error code is {finish_reason}")
    logger.info("Successfully generated image with the SDXL 1.0 model %s", model_id)
    return image_bytes

def save_image(image_bytes, output_path):
    """Save the image bytes to a file."""
    image = Image.open(io.BytesIO(image_bytes))
    image.save(output_path, format="JPEG", quality=95)
    logger.info(f"Image saved to {output_path}")

def adjust_image_size(image_path, target_size):
    """Adjusts the size of the image to the nearest multiple of 64."""
    with Image.open(image_path) as img:
        adjusted_width = ((target_size[0] // 64) + (1 if target_size[0] % 64 else 0)) * 64
        adjusted_height = ((target_size[1] // 64) + (1 if target_size[1] % 64 else 0)) * 64
        img = img.resize((adjusted_width, adjusted_height), Image.LANCZOS)
        img = img.crop((0, 0, target_size[0], target_size[1]))
        return img

def image_to_image_request(imageToImageRequest, source_image, generated_images):
    """Entrypoint for SDXL example."""
    image = Image.open(source_image)
    new_image = image.resize((imageToImageRequest.image_width, imageToImageRequest.image_height))

    # Save the resized image to an in-memory file
    buffered = io.BytesIO()
    new_image.save(buffered, format="JPEG")
    init_image = base64.b64encode(buffered.getvalue()).decode("utf8")

    body = json.dumps({
        "text_prompts": [
            {"text": imageToImageRequest.positive_prompt, "weight": 1},
            {"text": imageToImageRequest.negative_prompt, "weight": -1}
        ],
        "init_image": init_image,
        "init_image_mode": imageToImageRequest.init_image_mode,
        "image_strength": imageToImageRequest.image_strength,
        "cfg_scale": imageToImageRequest.cfg_scale,
        "clip_guidance_preset": imageToImageRequest.clip_guidance_preset,
        "sampler": imageToImageRequest.sampler,
        "samples": imageToImageRequest.samples,
        "seed": imageToImageRequest.seed,
        "steps": imageToImageRequest.steps,
        "style_preset": imageToImageRequest.style_preset
    })
    try:
        logger.info(f"Source image: {source_image}")
        image_bytes = generate_image_from_image(model_id=MODEL_ID, body=body)
        epoch_time = int(time.time())
        generated_image_path = f"{generated_images}/image_{epoch_time}_{imageToImageRequest.seed}_{imageToImageRequest.sampler}_{imageToImageRequest.image_strength}_{imageToImageRequest.cfg_scale}_{imageToImageRequest.steps}_{imageToImageRequest.style_preset}.jpg"
        save_image(image_bytes, generated_image_path)
    except ClientError as err:
        logger.error("A client error occurred: %s", err.response["Error"]["Message"])
    except ImageError as err:
        logger.error(err.message)
    else:
        logger.info(f"Finished generating image with SDXL model {MODEL_ID}.")

if __name__ == "__main__":
    POSITIVE_PROMPT = "Replace shoes with a clear and modern water bottle"
    NEGATIVE_PROMPT = "(worst quality:1.5), (low quality:1.5), (normal quality:1.5), low-res, bad anatomy, bad hands, watermark, moles, toes, (monochrome:1.5), (grayscale:1.5), (bad proportions:1.3)"

    seed = random.randrange(1, 1000000)
    seed = 8070

    imageToImageRequest = ImageToImageRequest(
        image_width=512,
        image_height=512,
        positive_prompt=POSITIVE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        init_image_mode="IMAGE_STRENGTH",
        image_strength=0.35,
        cfg_scale=7,
        clip_guidance_preset="NONE",
        sampler="K_DPMPP_2M",
        samples=1,
        seed=seed,
        steps=30,
        style_preset=StylesPresets.FANTASY_ART.value
    )

    f_filename = "generated_image_20240708235904.png"
    text_to_image_path = os.path.join(GENERATED_IMAGES, "text-image", f_filename) # 完整路徑
    image_to_image_path = os.path.join(GENERATED_IMAGES, "image-image")
    image_to_image_request(imageToImageRequest, text_to_image_path, image_to_image_path)

    # import random
    # for i in range(10):
    #     seed = random.randrange(1, 1000000)
    #     logger.info(f"Random seed: {seed}")
    #     imageToImageRequest.seed = seed
    #     image_to_image_request(imageToImageRequest, SOURCE_IMAGE, GENERATED_IMAGES)
