import asyncio

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.utils.settings import azure_openai_settings_from_dot_env_as_dict

from DallePlugin import Dalle3


def read_file(file_name):
    with open(file_name, mode='r', encoding='utf8') as file:
        data = file.read()
    return data

async def main():
    system_message = """
    You are a chatbot which will help with creating posts to LinkedIn.
    """
    
    kernel = sk.Kernel()

    service_id = "chat-gpt"
    chat_service = sk_oai.AzureChatCompletion(
        service_id=service_id, **azure_openai_settings_from_dot_env_as_dict(include_api_version=True)
    )

    kernel.add_service(chat_service)

    req_settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
    req_settings.max_tokens = 2000
    req_settings.temperature = 0.7
    req_settings.top_p = 0.8
    req_settings.auto_invoke_kernel_functions = True    

    presentation_text = read_file("presentation_text.txt")

    summarization_function = kernel.create_function_from_yaml(read_file("summarization_function.yaml"), "LinkedInPlugin")
    dalle3 = kernel.import_plugin_from_object(Dalle3(), "Dalle3")

    summarization = await kernel.invoke(
        functions=summarization_function, 
        presentation_text=presentation_text)

    animal_pic_url = await kernel.run_async(
        dalle3['ImageFromPrompt'],
        input_str=summarization
    )

    print(summarization)
    print(animal_pic_url)

if __name__ == "__main__":
    asyncio.run(main())