import custom_console
import asyncio

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool, QueryEngineTool
# Remembering state of the chat
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)

custom_console.clear_console()
custom_console.simple_spinner(duration=3)

def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

llm = GoogleGenAI(model_name="gemini-1.5-flash")

# as shown in the Components in LlamaIndex section
query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)

agent = AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(multiply)],
    llm=llm
)

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="name",
    description="a specific description",
    return_direct=False,
)

query_engine_agent = AgentWorkflow.from_tools_or_functions(
    [query_engine_tool],
    llm=llm,
    system_prompt="You are a helpful assistant that has access to a database containing persona descriptions. "
)

calculator_agent = ReActAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant. Use your tools for any math operation.",
    tools=[add, subtract],
    llm=llm,
)

query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information about XYZ",
    system_prompt="Use your tool to query a RAG system to answer information about XYZ",
    tools=[query_engine_tool],
    llm=llm
)

agent = AgentWorkflow(
    agents=[calculator_agent, query_agent], root_agent="calculator"
)


async def main():
    ctx = Context(agent)
    # response = await agent.run("My name is Bob.", ctx=ctx)
    # response = await agent.run("What was my name again?", ctx=ctx)
    response = await agent.run(user_msg="Can you add 5 and 3?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
