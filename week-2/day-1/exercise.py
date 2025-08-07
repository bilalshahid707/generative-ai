import os
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display, update_display

from google import genai
from google.genai import types


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

MODEL="gemini-2.5-flash"

gpt_system = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."

claude_system = "You are a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting."

gpt_msgs=["hi there"]
claude_msgs=["hi"]

def call_gpt():
    messages=[{"role":"system","content":gpt_system}]
    for gpt,claude in zip(gpt_msgs,claude_msgs):
        messages.append({"role":"assistant","content":gpt})
        messages.append({"role":"user","content":claude})
    response=openai.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    return response.choices[0].message.content

def call_claude():
    messages=[]
    for gpt,claude in zip(gpt_msgs,claude_msgs):
        messages.append({"role":"assistant","content":claude})
        messages.append({"role":"user","content":gpt})
    messages.append({"role":"user","content":gpt_msgs[-1]})
    response=openai.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    return response.choices[0].message.content

for i in range(5):
    
    gpt_next = call_gpt()
    print("from_gpt: ",gpt_next)
    gpt_msgs.append(gpt_next)

    claude_next = call_claude()
    print("from_claude: ",claude_next)
    claude_msgs.append(claude_next)