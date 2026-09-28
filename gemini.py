import os
from time import sleep

from dotenv import load_dotenv
from google import genai

load_dotenv()

google_api_key = os.environ["GOOGLE_STUDIO_API"]


client = genai.Client(api_key=google_api_key)


for i in range(10):
    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input="Explain how AI works in a few words"
    )
    print(interaction.output_text)
    sleep(1)
