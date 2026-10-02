from huggingface_hub import InferenceClient
import gradio as gr
import os


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

client = InferenceClient(
    model=MODEL_ID,
    provider="auto",
    token=os.environ.get("HF_TOKEN")
)


def format_prompt(message, history):
    messages = []

    for user_prompt, bot_response in history:
        messages.append({
            "role": "user",
            "content": user_prompt
        })
        messages.append({
            "role": "assistant",
            "content": bot_response
        })

    messages.append({
        "role": "user",
        "content": message
    })

    return messages


def generate(
    prompt,
    history,
    temperature=0.9,
    max_new_tokens=256,
    top_p=0.95,
    repetition_penalty=1.0,
):
    temperature = float(temperature)

    if temperature < 1e-2:
        temperature = 1e-2

    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=top_p,
        seed=42,
    )

    messages_for_api = format_prompt(prompt, history)

    stream = client.chat.completions.create(
        messages=messages_for_api,
        **generate_kwargs,
        stream=True,
    )

    output = ""

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            output += chunk.choices[0].delta.content
            yield output

    return output


additional_inputs = [
    gr.Slider(
        label="Temperature",
        value=0.9,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Higher values produce more diverse outputs",
    ),
    gr.Slider(
        label="Max new tokens",
        value=256,
        minimum=0,
        maximum=1048,
        step=64,
        interactive=True,
        info="The maximum numbers of new tokens",
    ),
    gr.Slider(
        label="Top-p (nucleus sampling)",
        value=0.90,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Higher values sample more low-probability tokens",
    ),
    gr.Slider(
        label="Repetition penalty",
        value=1.2,
        minimum=1.0,
        maximum=2.0,
        step=0.05,
        interactive=True,
        info="Penalize repeated tokens",
    )
]


css = """
  #mkd {
    height: 500px;
    overflow: auto;
    border: 1px solid #ccc;
  }
"""


with gr.Blocks(css=css) as demo:

    gr.HTML(
        "<h1><center>Qwen 2.5 7B Instruct</center></h1>"
    )

    gr.HTML(
        "<h3><center>"
        "In this demo, you can chat with "
        "<a href='https://huggingface.co/Qwen/Qwen2.5-7B-Instruct'>"
        "Qwen2.5-7B-Instruct"
        "</a> model. 💬"
        "</center></h3>"
    )

    gr.HTML(
        "<h3><center>"
        "Learn more about the model "
        "<a href='https://huggingface.co/Qwen/Qwen2.5-7B-Instruct'>"
        "here"
        "</a>. 📚"
        "</center></h3>"
    )

    gr.ChatInterface(
        generate,
        additional_inputs=additional_inputs,
        examples=[
            ["What is the secret to life?"],
            ["Write me a recipe for pancakes."]
        ]
    )


demo.queue(
    default_concurrency_limit=75,
    max_size=100
).launch(debug=True)
