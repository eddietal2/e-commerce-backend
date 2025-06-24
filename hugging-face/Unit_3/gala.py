# Import necessary libraries
import asyncio
from llama_index.core.agent.workflow import AgentWorkflow
import custom_console
from tools import weather_info_tool, hub_stats_tool
# from retriever import guest_info_tool

from llama_index.llms.google_genai import GoogleGenAI

def tool():
    print('This is a Tool!')

alfred = AgentWorkflow.from_tools_or_functions(
    [weather_info_tool, hub_stats_tool],
    llm=GoogleGenAI(model="gemini-1.5-flash", temperature=0.0),
)
async def main():
    query = "Tell me about Lady Ada Lovelace. What's her background?"
    response = await alfred.run(query)
    custom_console.clear_console()
    custom_console.simple_spinner(duration=3)
    print("🎩 Alfred's Response:")
    print(response.response.blocks[0].text)


if __name__ == "__main__":
    asyncio.run(main())
