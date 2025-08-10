import os
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display, update_display
import gradio as gr
from tavily import TavilyClient
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

system_message = """
You are a highly knowledgeable and accurate technical assistant. 
Your primary role is to provide answers that are **detailed, structured, and well-formatted**, 
while matching the user's level of expertise.

**Response Guidelines:**

1. **Depth & Completeness** – Always provide a thorough, multi-paragraph response. 
   - Give background context before answering directly.
   - Include explanations, examples, and relevant use cases.
   - Anticipate and answer follow-up questions proactively.

2. **Clarity & Structure** – Organize answers with:
   - Clear section headings
   - Numbered steps or bullet points
   - Code snippets (if applicable)
   - Tables or comparisons where helpful

3. **Formatting** – Use **bold** for emphasis, `inline code` for commands or function names, and fenced code blocks for multi-line code.

4. **Context Awareness** – Adapt explanations to the user’s skill level.
   - For beginners: simplify concepts with analogies.
   - For advanced users: use precise technical terminology.

5. **Examples & References** – Include:
   - Practical examples
   - Links to official documentation or trusted resources
   - Real-world scenarios

6. **Best Practices** – Always mention:
   - Common pitfalls
   - Optimization tips
   - Recommended approaches

7. **Accuracy First** – Verify that your explanations are factually correct and up-to-date.
   - If unsure, clearly state limitations and suggest reliable sources.

**Output Style:**
- Always return a **comprehensive, well-structured response**, never a one-liner.
- Assume the user wants the *full picture* unless they explicitly request a short answer.
"""


system_message += """
When deciding what to send to the `web_search` function:
- Parse the user’s request into the minimal set of separate search queries needed to fully answer it.
- Put each query as a separate string in the `queries` array.
- Avoid merging different search intents into a single query.
"""

client = TavilyClient("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")

def web_search(query):
    response = client.search(
        query=query
    )
    return response

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Executes a focused live web search to obtain accurate, current, and authoritative "
                "technical information from reliable sources. Use this function when the question depends "
                "on the latest data, documentation, or events that may have changed since the model’s training cutoff."
                "\n\n✅ Use when:"
                "\n1. Checking recent releases, API updates, or bug fixes in software libraries/frameworks."
                "\n2. Retrieving the latest official documentation, RFCs, or technical specs."
                "\n3. Looking up real-time service status, security advisories, or infrastructure metrics."
                "\n4. Gathering up-to-date industry news in software, AI/ML, or emerging technologies."
                "\n5. Finding newly published research, benchmarks, or open-source project updates."
                "\n\n❌ Avoid when:"
                "\n- Answering questions about stable concepts, algorithms, or historical facts."
                "\n- Performing code reasoning or logic tasks solvable without live search."
                "\n- Handling topics unlikely to have changed in recent years."
                "\n\n💡 Rule of thumb: If accuracy depends on the most recent developments, call this function."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "description": "The exact search queries to send to the search engine.",
                        "items": {  # <-- REQUIRED
                            "type": "string",
                            "description": "A single web search query string."
                        }
                    }
                },
                "required": ["queries"]
            }
        }
    }
]


TOOLS={
    "web_search":web_search
}

def handle_tool_call(chunk,reply):
    tool_list=chunk.choices[0].delta.tool_calls
    for tool in tool_list:
        function_name=tool.function.name
        arguments=json.loads(tool.function.arguments)

        queries = arguments.get("queries", [])
        
        for query in queries:
            tool_response = TOOLS[function_name](query)
            reply += tool_response["results"][0]["content"]

    return reply
    
def chat(message,history):
    messages=[{"role":"system","content":system_message}]+history+[{"role":"user","content":message}]
    stream = openai.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
        stream=True,
        tools=tools
    )

    reply=""
    for chunk in stream:
        if chunk.choices[0].finish_reason=="tool_calls":
            reply+=handle_tool_call(chunk,reply)
        reply+=chunk.choices[0].delta.content or ''
        yield reply
        
gr.ChatInterface(fn=chat,type="messages").launch()