from dotenv import load_dotenv
from openai import OpenAI

import requests
import os
import shutil

from typing_extensions import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function

class Dalle3:
    @kernel_function(
        description="Generates an with DALL-E 3 model based on a prompt",
        name="ImageFromPrompt",
    )

    def ImageFromPrompt(
        self,
        input: Annotated[str, "The prompt to generate the image."]
    ) -> Annotated[str, "The output link to the image"]:
        """Generates an image with DALL-E 3 model based on a prompt."""
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.images.generate(
            model="dall-e-3",
            prompt=input,
            size="1024x1024",
            quality="standard",
            n=1,
        )

        image_url = response.data[0].url

        imageContent = requests.get(image_url, stream=True)
        if imageContent.status_code == 200:
            with open("temp_image.png", "wb") as out_file:
                    shutil.copyfileobj(imageContent.raw, out_file)

        return image_url