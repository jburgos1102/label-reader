from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-5", input="Say hello to Joel and confirm the API is working."
)

print(response.output_text)
