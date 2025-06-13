import os
import custom_console

# https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
from huggingface_hub import InferenceClient
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

# https://huggingface.co/agents-course/notebooks/blob/main/unit1/dummy_agent_library.ipynb
# Resume @ In [6]