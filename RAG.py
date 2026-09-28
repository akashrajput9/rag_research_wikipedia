import os

import voyageai
from dotenv import load_dotenv

load_dotenv()

vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

result = vo.embed(["hello world"], model="voyage-4-large")

print(result)
