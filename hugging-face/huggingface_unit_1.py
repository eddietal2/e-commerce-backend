import os
import custom_console
from huggingface_hub import InferenceClient

# https://huggingface.co/agents-course/notebooks/blob/main/unit1/dummy_agent_library.ipynb
# https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct

HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(provider="hf-inference", model="meta-llama/Llama-3.3-70B-Instruct")
# if the outputs for next cells are wrong, the free model may be overloaded. You can also use this public endpoint that contains Llama-3.3-70B-Instruct
# client = InferenceClient("https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud")

# As seen in the LLM section, if we just do decoding, **the model will only stop when it predicts an EOS token**, 
# and this does not happen here because this is a conversational (chat) model and we didn't apply the chat template it expects.
prompt_one="""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
The Capital of Michigan is<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
prompt_two = "The Capital of Michigan is"
output_one = client.text_generation(
    prompt_one,
    max_new_tokens=100,
)
output_two = client.text_generation(
    prompt_two,
    max_new_tokens=100,
)

custom_console.clear_console()
custom_console.simple_spinner(duration=3)
print('\n')
print(f'{custom_console.COLOR_BLUE}Prompt: ', prompt_two)
print('\n')
print(f'{custom_console.COLOR_RED}With EOT{custom_console.RESET_COLOR}')
print(output_one,'\n')
print(f'{custom_console.COLOR_GREEN}Without EOT{custom_console.RESET_COLOR}')
print(output_two,'\n')

output_three = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "The Capital of Michigan is"}
    ],
    stream=False,
    max_tokens=1024,
)
# Chat Message via 'client.chat.completions.create'
print(output_three.choices[0].message.content)

# Dummy Agent
# This system prompt is a bit more complex and actually contains the function description already appended.
# Here we suppose that the textual description of the tools have already been appended
SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {{"location": {{"type": "string"}}}}
example use :
```
{{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}}

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:
```
$JSON_BLOB
```
Observation: the result of the action. This Observation is unique, complete, and the source of truth.
... (this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. """

# Since we are running the "text_generation", we need to add the right special tokens.
prompt_three=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT}
<|eot_id|><|start_header_id|>user<|end_header_id|>
What's the weather in London ?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

output_four = client.text_generation(
    prompt_three,
    max_new_tokens=150,
    stop=["Observation:"]
)
# Dummy function
def get_weather(location):
    """This function is an AI Agent Tool"""
    return f"the weather in {location} is sunny with low temperatures. \n"

get_weather('London')
prompt_four = prompt_three+output_four+get_weather('London')
print(prompt_four)

final_output = client.text_generation(
    prompt_four,
    max_new_tokens=200,
)

print(final_output)