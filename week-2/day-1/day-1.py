import os
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display, update_display

load_dotenv(override=True)
gemini_api_key = os.getenv('GEMINI_API_KEY')

if gemini_api_key:
    print(f"Google API Key exists and begins {gemini_api_key[:8]}")
else:
    print("Google API Key not set")

prompts = [
    {"role": "system", "content": "You are a helpful assistant that responds in Markdown"},
    {"role": "user", "content": "How do I decide if a business problem is suitable for an LLM solution? Please respond in Markdown."}
]

# Using open ai
openai = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Streaming response
stream = openai.chat.completions.create(
    model="gemini-2.5-flash",
    messages=prompts,
    stream=True
)
reply = ""
display_handle = display(Markdown(""), display_id=True)
for chunk in stream:
    reply += chunk.choices[0].delta.content or ''
    reply = reply.replace("```","").replace("markdown","")
    update_display(Markdown(reply), display_id=display_handle.display_id)

# Without streaming
response = openai.chat.completions.create(
    model="gemini-2.5-flash",
    messages=prompts,
)
display(Markdown(response.choices[0].message.content))







