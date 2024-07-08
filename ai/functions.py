import os

from openai import OpenAI

client = OpenAI(
    # api_key=os.getenv('OPENAI_API_KEY'),  # this is also the default, it can be omitted
)

DEFAULT_MODEL='gpt-4o'

def openai_call(messages, model: str = DEFAULT_MODEL, max_tokens: int = None, temperature: float = 0, presence_penalty: float = 0):
    # add system message
    # messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + messages
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        presence_penalty=presence_penalty,
    )

    return completion.choices[0].message.content