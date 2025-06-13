import os
import sys
import time
# https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
from huggingface_hub import InferenceClient
HF_TOKEN = os.environ.get("HF_TOKEN")

# ANSI escape codes for colors, etc, in print statements
COLOR_RED = "\033[91m"   # Bright Red
COLOR_GREEN = "\033[92m" # Bright Green
RESET_COLOR = "\033[0m"

def simple_spinner(duration=5):
    spinner_chars = ['-', '\\', '|', '/'] # Characters for the spinner
    start_time = time.time()
    i = 0
    while time.time() - start_time < duration:
        sys.stdout.write(f'\rLoading {spinner_chars[i % len(spinner_chars)]}')
        sys.stdout.flush()
        time.sleep(0.1) # Controls the speed of the spin
        i += 1
    sys.stdout.write('\rLoading complete!  \n') # Overwrite with final message and a newline
    sys.stdout.flush()

# Clear console log
def clear_console():
    """Clears the console screen based on the operating system."""
    os.system('cls' if os.name == 'nt' else 'clear')

client = InferenceClient(provider="hf-inference", model="meta-llama/Llama-3.3-70B-Instruct")
# if the outputs for next cells are wrong, the free model may be overloaded. You can also use this public endpoint that contains Llama-3.3-70B-Instruct
# client = InferenceClient("https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud")

# As seen in the LLM section, if we just do decoding, **the model will only stop when it predicts an EOS token**, 
# and this does not happen here because this is a conversational (chat) model and we didn't apply the chat template it expects.
prompt="""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
The Capital of Michigan is<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
output_one = client.text_generation(
    prompt,
    max_new_tokens=100,
)
output_two = client.text_generation(
    "The Capital of Michigan is",
    max_new_tokens=100,
)

clear_console()
simple_spinner(duration=3)
print(f'{COLOR_RED}With EOT{RESET_COLOR}')
print(output_one,'\n')
print(f'{COLOR_GREEN}Without EOT{RESET_COLOR}')
print(output_two,'\n')

# https://huggingface.co/agents-course/notebooks/blob/main/unit1/dummy_agent_library.ipynb
# Resume @ In [6]