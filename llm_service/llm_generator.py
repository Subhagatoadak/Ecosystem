import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Retrieve API keys
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


def generate_llm_response(prompt, model="gpt-4o", temperature=0.7):
    """
    Generates a response from OpenAI's GPT-4 model given a prompt.
    The API key is expected to be set in the OPENAI_API_KEY environment variable.
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}"
