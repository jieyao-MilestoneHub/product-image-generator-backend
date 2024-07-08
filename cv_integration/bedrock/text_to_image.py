import os
import base64
import io
import json
import logging
from datetime import datetime
import random
from PIL import Image
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

# AWS Credentials
AWS_KEY_ID = os.getenv("AWS_KEY_ID")
AWS_KEY_SECRET = os.getenv("AWS_KEY_SECRET")

# Constants
MODEL_ID = "stability.stable-diffusion-xl-v1"

class ImageError(Exception):
    """
    Custom exception for errors returned by SDXL 1.0.
    """

    def __init__(self, message):
        self.message = message


def get_bedrock_client():
    """Initialize and return the Bedrock client."""
    return boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        aws_access_key_id=AWS_KEY_ID,
        aws_secret_access_key=AWS_KEY_SECRET
    )


def generate_image_from_text(model_id, body):
    """
    Generate an image using SDXL 1.0 on demand.
    
    Args:
        model_id (str): The model ID to use.
        body (str) : The request body to use.
    
    Returns:
        image_bytes (bytes): The image generated by the model.
    """
    logging.info("Generating image with SDXL model %s", model_id)

    bedrock = get_bedrock_client()

    response = bedrock.invoke_model(
        body=body, 
        modelId=model_id, 
        accept="application/json", 
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    logging.info(f"Bedrock result: {response_body['result']}")

    artifact = response_body.get("artifacts")[0]
    base64_image = artifact.get("base64")
    image_bytes = base64.b64decode(base64_image.encode("ascii"))

    finish_reason = artifact.get("finishReason")
    if finish_reason in ["ERROR", "CONTENT_FILTERED"]:
        raise ImageError(f"Image generation error. Error code is {finish_reason}")

    logging.info("Successfully generated image with the SDXL 1.0 model %s", model_id)
    return image_bytes


def save_image(image_object, output_path, adjust=False):
    """
    Save the adjusted image to the specified path.
    """
    if not adjust:
        # Convert bytes to an Image object
        image_bytes = image_object
        image = Image.open(io.BytesIO(image_bytes))
        image.save(output_path, format="JPEG", quality=95)
        logging.info(f"Generated image saved to {output_path}")
    else:
        image = image_object
        image.save(output_path, format="JPEG", quality=95)
        logging.info(f"Image saved to {output_path}")

def text_to_image_request(model_id, positive_prompt, negative_prompt, seed, generated_image_path):
    """
    Entrypoint for SDXL example.
    
    Args:
        model_id (str): The model ID to use.
        positive_prompt (str): The positive prompt to use.
        negative_prompt (str): The negative prompt to use.
    """
    # Build request body
    body = json.dumps({
        "text_prompts": [
            {"text": positive_prompt, "weight": 1},
            {"text": negative_prompt, "weight": -1}
        ],
        "height": 512,
        "width": 512,
        "cfg_scale": 12,
        "clip_guidance_preset": "NONE",
        "sampler": "K_DPMPP_2M",
        "samples": 1,
        "seed": seed,
        "steps": 25,
        "style_preset": "fantasy-art"
    })

    # Generate and save image
    try:
        image_bytes = generate_image_from_text(model_id=model_id, body=body)
        save_image(image_bytes, generated_image_path, adjust=False)
    except ClientError as err:
        logging.error(f"A client error occurred: {err.response['Error']['Message']}")
    except ImageError as err:
        logging.error(err.message)
    finally:
        logging.info(f"Finished generating image with SDXL model {model_id}.")

def adjust_image_size(image_path, target_size):
    """
    Adjusts the size of the image to the nearest multiple of 64.
    """
    with Image.open(image_path) as img:
        # Calculate the nearest size multiple of 64
        adjusted_width = ((target_size[0] // 64) + (1 if target_size[0] % 64 else 0)) * 64
        adjusted_height = ((target_size[1] // 64) + (1 if target_size[1] % 64 else 0)) * 64

        # Resize image to the adjusted size using LANCZOS (formerly ANTIALIAS)
        img = img.resize((adjusted_width, adjusted_height), Image.LANCZOS)

        # Optionally, crop the image back to the desired size if necessary
        img = img.crop((0, 0, target_size[0], target_size[1]))

        return img

if __name__ == "__main__":

    interest = "sports"
    product_name = "kettle"
    product_feature = "【Blender Bottle】〈Strada Stainless Steel〉Push-type leak-proof shaker cup 710ml 'originally importe' (BlenderBottle/Sports Bottle/Ice Cup)"
    age = "25-34"
    gender = "male"

    POSITIVE_PROMPT = (
        f"Generate a clean and clear advertisement background that is ideal for {interest}. "
        f"The image should not contain any animals or brand logos. "
        f"Focus on creating an atmospheric scene of {interest}, utilizing the rule of thirds to craft a high-quality venue. "
        f"Do not include {product_name} or any similar items in the image. "
        f"Design the scene to appeal to {age} year old {gender} individuals, ensuring no specific cultural symbols or text are present. "
        f"The purpose of this advertisement is to highlight the {product_feature} of the {product_name}. "
        f"Create a setting that is inviting and engaging for the target audience, "
        f"with a clean, minimalistic, and modern style. "
        f"Use soft lighting, natural elements, and open spaces to enhance the appeal."
    )

    NEGATIVE_PROMPT = (
        "(worst quality:1.5), (low quality:1.5), (normal quality:1.5), "
        "lowres, bad anatomy, bad hands, watermark, moles, toes, bad-picture-chill-75v, "
        "realisticvision-negative-embedding, (monochrome:1.5), (grayscale:1.5), "
        "(bad proportions:1.3), animals, brand logos, specific cultural symbols, text, "
        "cluttered backgrounds, harsh lighting, overly dark scenes, and any elements that distract from the clean, clear, and modern aesthetic."
    )

    POSITIVE_PROMPT = (
        f"Craft a clear and realistic image that appeals to a {age}-year-old {gender} interested in {interest}. "
        "The setting should be minimalistic and modern, perfect for subtle product placement. "
        "Ensure the scene is well-lit, using soft, natural light to enhance the sophisticated atmosphere. "
        "The composition should follow the rule of thirds to create a visually appealing background that is free of animals, humans, brand logos, and the specific product. "
        "No text or cultural symbols should be visible. "
        "The overall look should convey cleanliness and simplicity, inviting the viewer to focus on the space designed for the product."
    )

    NEGATIVE_PROMPT = (
        "Avoid producing images with low resolution, incorrect anatomical features, or any form of clutter. "
        "Exclude watermarks, moles, and any specific body parts like toes. "
        "Do not incorporate harsh or overly dark lighting. "
        "The image should not contain any monochrome or grayscale elements, nor should it have disproportional features. "
        "Ensure there are no animals, brand logos, specific cultural symbols, or text. "
        "Refrain from creating backgrounds that are overly detailed or distract from a clean, modern aesthetic."
    )

    POSITIVE_PROMPT = (
        f"Generate a clean and clear advertisement background that is ideal for {interest}. "
        f"Focus on creating an atmospheric scene of {interest}, utilizing the rule of thirds to craft a high-quality venue. "
        f"Design the scene to appeal to {age}-year-old {gender} individuals, ensuring no specific cultural symbols or text are present. "
        f"The purpose of this advertisement is to highlight the {product_feature} of the {product_name}. "
        "Create a setting that is inviting and engaging for the target audience, with a clean, minimalistic, and modern style. "
        "Use soft lighting, natural elements, and open spaces to enhance the appeal."
    )

    NEGATIVE_PROMPT = (
        "Avoid producing images of low quality, with poor resolution or incorrect anatomy. "
        f"Exclude any representations of animals, brand logos, specific cultural symbols, text, and especially {product_name} or any similar items. "
        "Refrain from using monochrome, grayscale, and any elements that contribute to a cluttered or overly dark scene. "
        "Ensure the background remains uncluttered, focusing on maintaining a clean, clear, and modern aesthetic with no distractions that detract from the desired atmosphere."
    )

    POSITIVE_PROMPT = (
        f"Generate a clean and clear advertisement background that is ideal for {interest}. "
        f"Focus on creating an atmospheric scene of {interest}, utilizing the rule of thirds to craft a high-quality venue. "
        f"Design the scene to appeal to {age}-year-old {gender} individuals, ensuring no specific cultural symbols or text are present. "
        f"The purpose of this advertisement is to highlight the {product_feature} of the {product_name}. "
        "Create a setting that is inviting and engaging for the target audience, with a clean, minimalistic, and modern style. "
        "Use soft lighting, natural elements, and open spaces to enhance the appeal, specifically avoiding any products related to the interest, focusing on the venue itself."
    )

    NEGATIVE_PROMPT = (
        "Avoid producing images of low quality, with poor resolution or incorrect anatomy. "
        f"Exclude any representations of animals, brand logos, specific cultural symbols, text, and particularly any products or items related to {interest}, including {product_name}. "
        "Do not include any elements that suggest or resemble products associated with the interest. "
        "Refrain from using monochrome, grayscale, and any elements that contribute to a cluttered or overly dark scene. "
        "Ensure the background remains uncluttered, focusing on maintaining a clean, clear, and modern aesthetic with no distractions that detract from the desired atmosphere."
    )
    
    POSITIVE_PROMPT = (
        f"Generate a clean and clear advertisement background that is ideal for {interest}. "
        f"Focus on creating an atmospheric scene of {interest}, utilizing the rule of thirds to craft a high-quality venue suitable for product placement. "
        f"Design the scene to appeal to {age} year-old {gender} individuals, ensuring no specific cultural symbols or text are present. "
        f"The purpose of this advertisement is to subtly suggest a setting where the product feature, such as {product_feature}, could be highlighted, without showing the product itself. "
        "Create an environment that is inviting and engaging for the target audience, with a clean, minimalistic, and modern style. "
        "Use soft lighting, natural elements, and open spaces to enhance the appeal, ensuring the space remains ready for subtle product placement."
    )

    NEGATIVE_PROMPT = (
        "Avoid producing images of low quality, with poor resolution or incorrect anatomy. "
        f"Exclude any representations of animals, brand logos, specific cultural symbols, text, and particularly any products or items related to {interest}, including {product_name} or any similar items. "
        f"Do not include any elements that suggest or resemble products associated with {interest}. "
        "Refrain from using monochrome, grayscale, and any elements that contribute to a cluttered or overly dark scene. "
        "Ensure the background remains uncluttered, focusing on maintaining a clean, clear, and modern aesthetic with no distractions that detract from the desired atmosphere, perfect for product placement."
    )

    # AWS example
    POSITIVE_PROMPT = """a young woman in white awsshirt,
    masterpiece, clear face, cute, beautiful woman,
    normal eyes, front, clear face, beautiful face,
    elegant face, Intricate, High Detail, Sharp
    focus, professional portrait photograph,
    background in the studio, background is
    indoor, lightly smile, nature skin, fair skin,
    realistic, intricate, depth of field, f/1. 8, 85mm,
    medium shot, hdr, 8k, highres, modelshoot
    style"""

    NEGATIVE_PROMPT = """lowres, worst quality,
    (ugly:1.3), (disfigured,mutated hands,)
    (misshapen hands), (mutated fingers), (fused
    fingers), cross eyes"""

    # First combination
    interest = "sports"
    gender = "male"
    user_age_group = "25-34"
    information_level = "detailed"
    content_direction = "healthy living"
    advertising_material_production_direction = "brand awareness"
    brightness = "bright"
    hue = "fresh green"
    saturation = "vibrant"

    # Second combination
    # interest = "outdoor travel"
    # gender = "female"
    # user_age_group = "35-44"
    # information_level = "in-depth"
    # content_direction = "nature conservation"
    # advertising_material_production_direction = "promoting eco-friendly products"
    # brightness = "bright"
    # hue = "forest green"
    # saturation = "saturated"

    # Define the positive prompt
    POSITIVE_PROMPT = (
        f"The background of ads Photography of {information_level} where to place ads"
        f"{interest}, {gender}, {user_age_group}, {content_direction}, {advertising_material_production_direction}, {brightness}, {hue}, {saturation}, "
        "high resolution, high quality, professional color grading, clear shadows and highlights, atmospheric"
    )

    # Define the negative prompt
    NEGATIVE_PROMPT = (
        "low resolution, bad quality, ugly, blur, mutation, extra arms, extra legs, 3d, painting, "
        "cannot print any main characters (focus solely on the background), the image should not be overcrowded"
    )

    generated_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    generated_image_path=f"test_generated_image_{generated_timestamp}.png"
    seed = random.randrange(1, 1000000)
    seed = 8070
    text_to_image_request(MODEL_ID, POSITIVE_PROMPT, NEGATIVE_PROMPT, seed, generated_image_path)