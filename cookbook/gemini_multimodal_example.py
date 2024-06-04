# pip install google-generativeai

from beyondllm.llms import GeminiMultiModal
from PIL import Image
import os
from getpass import getpass

os.environ['GOOGLE_API_KEY'] = getpass("Enter your API key:")

"""
## Approach - 1 Using Image URL
"""

import requests
from io import BytesIO

url = 'https://akm-img-a-in.tosshub.com/indiatoday/images/story/202404/virat-kohli-281539841-16x9_1.jpg?VersionId=buqK4ik3eMrlZGF6BCUOo4EeB2xmx0xX&size=690:388'
response = requests.get(url)
if response.status_code == 200:
    img = Image.open(BytesIO(response.content))
else:
    print("Error loading the image")

"""
>>> Approach - 2
>>> or Approach - 2 Using Image File Locally
img = Image.open("./sample.jpg")
"""

llm = GeminiMultiModal()
print(llm.predict(prompt="name the wife of the person?",image=img))
