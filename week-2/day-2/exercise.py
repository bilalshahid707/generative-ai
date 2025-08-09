import os
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display, update_display
import gradio as gr
from Website import Website
import json
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

link_system_prompt = "You are provided with a list of links found on a webpage. \
You are able to decide which of the links would be most relevant to include in a brochure about the company, \
such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
link_system_prompt += "You should respond in JSON as in this example:"
link_system_prompt += """
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""

def get_link_user_prompt(website):
    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt

def get_links(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_link_user_prompt(website)}
      ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    return json.loads(result)

def get_details(url):
    website=Website(url)
    contents = website.get_contents()
    result="Landing Page\n"
    result+=contents
    links=get_links(url)
    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        result += Website(link["url"]).get_contents()
    return result

system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
Include details of company culture, customers and careers/jobs if you have the information."

def get_brochure_user_prompt(company_name, url):
    user_prompt = f"You are looking at a company called: {company_name}\n"
    user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
    user_prompt+=get_details(url)
    return user_prompt

def stream_brochure(company_name,url):
    website = Website(url)
    stream = openai.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role":"user","content":get_brochure_user_prompt(company_name,url)},
            {"role":"system","content":system_prompt}
        ],
        stream=True
    )
    result=""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ''
        result = result.replace("```","").replace("markdown","")
        yield result

view = gr.Interface(
    fn=stream_brochure,
    inputs=[gr.Textbox(label="Enter website name",placeholder="Hugging Face"),gr.Textbox(label="Type Url",placeholder="www.huggingface.com")],
    outputs=[gr.Markdown(label="Brochure")]
)
view.launch()
