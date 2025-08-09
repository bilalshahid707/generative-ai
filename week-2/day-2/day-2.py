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

#################################################### Day 2 #########################################################

# Streaming Response 
def stream_response(prompt):
    stream = openai.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role":"system","content":"You are a helpful assistant that responds in Markdown"},
            {"role":"user","content":prompt}
        ],
        stream=True
    )
    reply = ""
    for chunk in stream:
        reply += chunk.choices[0].delta.content or ''
        reply = reply.replace("```","").replace("markdown","")
        yield reply



# Without Streaming 
def generate_response(prompt):
    response = openai.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role":"system","content":"You are a helpful assistant that responds in Markdown"},
            {"role":"user","content":prompt}
        ],
    )
    return response.choices[0].message.content

# Creating interface
view = gr.Interface(
    fn=generate_response,
    inputs=[gr.Textbox(label="Enter your prompt", placeholder="Type your question here...")],
    outputs=[gr.Markdown(label="LLM Response")],
    title="LLM Response Generator",
)

view.launch(share=True)





