import asyncio

import ssl
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
)
from semantic_kernel.planners.basic_planner import BasicPlanner

from DallePlugin import Dalle3
from LinkedInPlugin import LinkedInPlugin

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
            ai_model_id="gpt-4-turbo-preview",
            api_key=api_key,
        ),
    )

    presentation_text = read_file("presentation_text.txt")


    kernel.create_function_from_yaml(read_file("summarization_function.yaml"), "TextPrompt")
    kernel.create_function_from_yaml(read_file("image_prompt_function.yaml"), "TextPrompt")
    kernel.import_plugin_from_object(LinkedInPlugin(), "LinkedInPlugin")
    kernel.import_plugin_from_object(Dalle3(), "Dalle3")

    question = "Generate a post for LinkedIn based on the following text, make sure to summarize the presentation text. Also include relevant image generated by Dalle, make sure to provide dalle with precise prompt." + presentation_text

    planner = BasicPlanner(service_id)
    basic_plan = await planner.create_plan(question, kernel)

    print(basic_plan.generated_plan)

    results = await planner.execute_plan(basic_plan, kernel)
if __name__ == "__main__":
    asyncio.run(main())