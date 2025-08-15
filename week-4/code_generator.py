import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv(override=True)
gemini_api_key = os.getenv('GEMINI_API_KEY')

openai = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
MODEL = 'gemini-2.5-flash'

system_message = "You are an assistant who reimplements Python code in high-performance C++ for PC."
system_message += "Respond only with C++ code. Please don't use triple backticks or language tags. Keep comments minimal and only where necessary."
system_message += "The C++ code must produce identical output to the original Python code in the fastest possible execution time."

def user_prompt_for(python):
    user_prompt = "Rewrite this Python code in C++ with the fastest possible implementation that produces identical output in the least time. "
    user_prompt += "Respond only with C++ code; do not explain your work other than a few comments. "
    user_prompt += "Pay attention to number types to ensure no int overflows. Remember to #include all necessary C++ packages such as iomanip.\n\n"
    user_prompt += python
    return user_prompt

def messages_for(python):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt_for(python)}
    ]

def stream_gemini(python):    
    stream = openai.chat.completions.create(model=MODEL, messages=messages_for(python), stream=True)
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        yield reply.replace('```cpp\n','').replace('```','')

with gr.Blocks() as ui:
    with gr.Row():
        python = gr.Textbox(label="Python code:")
        cpp = gr.Code(label="C++ code:")
    with gr.Row():
        convert = gr.Button("Convert code")

    convert.click(stream_gemini, inputs=[python], outputs=[cpp])

ui.launch(inbrowser=True)