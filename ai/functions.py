import asyncio

from openai import OpenAI

client = OpenAI()

DEFAULT_MODEL='gpt-4o'

async def openai_call(
    messages,
    model: str = DEFAULT_MODEL,
    max_tokens: int = None,
    temperature: float = 0,
    presence_penalty: float = 0,
    non_blocking: bool = True,
):
    # add system message
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        presence_penalty=presence_penalty,
        stream=True
    )

    # initialise response
    deltas = []
    for chunk in completion:
        delta = chunk.choices[0].delta
        if delta.content is not None:
            deltas.append(delta.content)

        if non_blocking:
            await asyncio.sleep(0) # yield event loop

    return ''.join(deltas)