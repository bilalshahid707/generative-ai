import os
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display, update_display
import gradio as gr

######################### Setup ####################################
 
load_dotenv(override=True)
gemini_api_key = os.getenv('GEMINI_API_KEY')

if gemini_api_key:
    print(f"Google API Key exists and begins {gemini_api_key[:8]}")
else:
    print("Google API Key not set")

openai = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

#################################################### Day 3 #########################################################

system_message = """You are a friendly and persuasive assistant in a clothing store.
Your goal is to help customers find items they’ll love, while gently highlighting our current sales event:

- Hats are 60% off.
- Most other items are 50% off.

Guidelines:

1. If the customer mentions a hat (e.g., "I'm looking to buy a hat"), enthusiastically point out the variety available and mention the 60% discount.
   Example: "Wonderful! We have a great selection of hats, and several are part of our sales event at 60% off!"

2. If the customer is unsure about what to buy, suggest they consider hats first because of the higher discount.

3. Maintain a warm, helpful, and conversational tone.

4. Throughout the conversation, subtly encourage adding more items to their purchase by mentioning the 50% off sale on most products.

5. At the end of the interaction, display the entire conversation history in a nicely formatted transcript so the customer can review it.
"""


def chat(message,history):
    messages=[
        [{"role":"system","content":system_message}] + history + [{"role":"user","content":message}]
    ]

    stream = openai.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        stream=True
    )

    reply=""
    for chunk in stream:
        reply+=chunk.choices[0].delta.content or ''
        yield reply

gr.ChatInterface(fn=chat,type="messages",title="Customer assistant").launch()