import asyncio

import ssl
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.utils.settings import azure_openai_settings_from_dot_env_as_dict
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
)

from DallePlugin import Dalle3

ssl.SSLContext.verify_mode = ssl.VerifyMode.CERT_OPTIONAL

def read_file(file_name):
    with open(file_name, mode='r', encoding='utf8') as file:
        data = file.read()
    return data

async def main():  
    kernel = sk.Kernel()

    service_id = "gpt"
    api_key, _ = sk.openai_settings_from_dot_env()
    kernel.add_service(
        OpenAIChatCompletion(
            service_id=service_id,
            ai_model_id="gpt-3.5-turbo-1106",
            api_key=api_key,
        ),
    )

    presentation_text = read_file("presentation_text.txt")

    summarization_function = kernel.create_function_from_yaml(read_file("summarization_function.yaml"), "LinkedInPlugin")
    dalle3 = kernel.import_plugin_from_object(Dalle3(), "Dalle3")

    summarization = await kernel.invoke(
        functions=summarization_function, 
        presentation_text=presentation_text)

    # animal_pic_url = await kernel.invoke(
    #     dalle3['ImageFromPrompt'],
    #     input=summarization
    # )

    print(summarization)
    # print(animal_pic_url)

if __name__ == "__main__":
    asyncio.run(main())