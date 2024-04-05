import os
import requests
import json

from typing_extensions import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function

class LinkedInPlugin:
    @kernel_function(
        description="Create a post on LinkedIn",
        name="LinkedInPlugin",
    )

    def UploadPostOnLinkedIn(
        self,
        postContent: Annotated[str, "Short post text content."],
        imageUrl: Annotated[str, "The image url to be added to the post."]
    ) -> None:
        """Creates a post on LinkedIn with the given text and image."""

        def initialize_image_upload(access_token, person_id) -> str:
            """Initializes an image upload on LinkedIn."""

            url = "https://api.linkedin.com/rest/images?action=initializeUpload"
            headers = {
                "LinkedIn-Version": "202304",
                "X-Restli-Protocol-Version": "2.0.0",
                "Content-Type": "text/plain",
                "Authorization": f"Bearer {access_token}"
            }

            data = {
                "initializeUploadRequest": {
                    "owner": f"urn:li:person:{person_id}"
                }
            }

            response = requests.post(url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                return response
            else:
                return "Error: " + response.text
        
        def upload_image(upload_url: str) -> str:
            """Downloads an image and uploads it to LinkedIn."""
            # Upload the image
            with open("temp_image.png", "rb") as image_file:
                files = {"file": image_file}
                response = requests.put(upload_url, files=files)
            # Delete the temporary image file
            os.remove("temp_image.png")
            if response.status_code == 200:
                return "Success: Image uploaded."
            else:
                return "Error: " + response.text
        
        def create_post(access_token: str, person_id: str, postContent: str, imageId: str) -> None:
            """Creates a post on LinkedIn."""

            url = "https://api.linkedin.com/rest/posts"

            headers = {
                "LinkedIn-Version": "202304",
                "X-Restli-Protocol-Version": "2.0.0",
                "Content-Type": "text/plain",
                "Authorization": f"Bearer {access_token}"
            }

            data = {
                "author": f"urn:li:person:{person_id}",
                "commentary": postContent,
                "visibility": "PUBLIC",
                "distribution": {
                    "feedDistribution": "MAIN_FEED",
                    "targetEntities": [],
                    "thirdPartyDistributionChannels": []
                },
                "content":{
                    "media": {
                        "id": imageId
                    }
                },
                "lifecycleState": "PUBLISHED"
            }

            response = requests.post(url, headers=headers, data=json.dumps(data))

            print(response.status_code)

        access_token = os.getenv("LINKEDIN_ACCESS_TOKEN")
        person_id = os.getenv("LINKEDIN_PERSON_ID")

        response = initialize_image_upload(access_token, person_id)
        responseJson = json.loads(response.text)
        upload_url = responseJson["value"]["uploadUrl"]
        image_id = responseJson["value"]["image"]
        upload_image(upload_url)

        create_post(access_token, person_id, postContent, image_id)
        
    