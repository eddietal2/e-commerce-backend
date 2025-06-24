from llama_index.core.agent.workflow import AgentWorkflow # type: ignore
from llama_index.llms.google_genai import GoogleGenAI # type: ignore
from tools import guest_dataset
from retriever import guest_info_tool, get_guest_info_retriever
import custom_console

# Initialize the Hugging Face model
llm = GoogleGenAI(model="gemini-1.5-flash", temperature=0.0)

# Create Alfred, our gala agent, with the guest info tool
alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool],
    llm=llm,
)

async def main():
    # Example query Alfred might receive during the gala
    response = await alfred.run("Tell me about our guest named 'Lady Ada Lovelace'.")
    custom_console.clear_console()
    custom_console.simple_spinner(duration=3)
    print("🎩 Alfred's Response:")  
    print(response)