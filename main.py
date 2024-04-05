import asyncio
import os
import ssl
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
)

from semantic_kernel.planners import SequentialPlanner

from DallePlugin import Dalle3
from LinkedInPlugin import LinkedInPlugin

ssl.SSLContext.verify_mode = ssl.VerifyMode.CERT_OPTIONAL

def read_file(file_name, *args):
    full_file_path = os.path.join(*args, file_name)
    with open(full_file_path, mode='r', encoding='utf8') as file:
        data = file.read()
    return data

async def main():  
    kernel = sk.Kernel()

    service_id = "planner"
    api_key, _ = sk.openai_settings_from_dot_env()
    kernel.add_service(
        OpenAIChatCompletion(
            service_id=service_id,
            ai_model_id="gpt-4",
            api_key=api_key,
        ),
    )

    presentation_text = read_file("presentation_text.txt", "resources")

    kernel.create_function_from_yaml(read_file("image_prompt_function.yaml", "prompts"), "TextPrompt")
    kernel.create_function_from_yaml(read_file("summarization_function.yaml", "prompts"), "TextPrompt")
    kernel.import_plugin_from_object(LinkedInPlugin(), "LinkedInPlugin")
    kernel.import_plugin_from_object(Dalle3(), "Dalle3")

    question = f"""
    Generate a post for LinkedIn based on the presentation text provided below. 
    Make sure to include text to the post.
    Make sure to include image to the text.
    If you generate image with Dalle3 make sure to give it price prompt.
    Make sure to create upload the post on LinkedIn.

    ```presentation-text
     {presentation_text}
    ```
    """

    planner = SequentialPlanner(kernel, service_id)

    sequential_plan = await planner.create_plan(goal=question)

    kernel.on_function_invoking = lambda function: print(f"Invoking function: {function.name}")

    for step in sequential_plan._steps:
        print(step.description, ":", step._state.__dict__)
    
    await sequential_plan.invoke(kernel)

if __name__ == "__main__":
    asyncio.run(main())