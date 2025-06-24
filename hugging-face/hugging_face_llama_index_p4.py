import custom_console
import asyncio
from llama_index.core.workflow import Context, StartEvent, StopEvent
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.llms.google_genai import GoogleGenAI

# Tools
async def add(ctx: Context, a: int, b: int) -> int:
    """Add two numbers."""
    # update our count
    cur_state = await ctx.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.set("state", cur_state)

    return a + b

async def multiply(ctx: Context, a: int, b: int) -> int:
    """Multiply two numbers."""
    # update our count
    cur_state = await ctx.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.set("state", cur_state)

    return a * b

llm = GoogleGenAI(model_name="gemini-1.5-flash")

# Agents
multiply_agent = ReActAgent(
    name="multiply_agent",
    description="Is able to multiply two integers",
    system_prompt="A helpful assistant that can use a tool to multiply numbers.",
    tools=[multiply],
    llm=llm,
)

addition_agent = ReActAgent(
    name="add_agent",
    description="Is able to add two integers",
    system_prompt="A helpful assistant that can use a tool to add numbers.",
    tools=[add],
    llm=llm,
)

# Workflow
workflow = AgentWorkflow(
    agents=[multiply_agent, addition_agent],
    root_agent="multiply_agent",
    initial_state={"num_fn_calls": 0},
    state_prompt="Current state: {state}. User message: {msg}"
)

# Run the system
async def main():
    custom_console.clear_console()
    custom_console.simple_spinner(duration=3)
    ctx = Context(workflow)
    response1 = await workflow.run(user_msg="Can you add 5 and 3?", ctx=ctx)
    response2 = await workflow.run(user_msg="What was the last equation you solved?", ctx=ctx)
    state = await ctx.get("state")
    print(response1)    
    print(response2)    
    print(state["num_fn_calls"])    

if __name__ == "__main__":
    asyncio.run(main())
