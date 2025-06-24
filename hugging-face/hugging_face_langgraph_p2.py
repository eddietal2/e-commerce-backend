import base64
import custom_console
from typing import List, TypedDict, Annotated, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display

class AgentState(TypedDict):
    # The document provided
    input_file: Optional[str]  # Contains file path (PDF/PNG)
    messages: Annotated[list[AnyMessage], add_messages]

vision_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

def extract_text(img_path: str) -> str:
    """
    Extract text from an image file using a multimodal model.
    
    Master Wayne often leaves notes with his training regimen or meal plans.
    This allows me to properly analyze the contents.
    """
    all_text = ""
    try:
        # Read image and encode as base64
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Prepare the prompt including the base64 image data
        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Extract all the text from this image. "
                            "Return only the extracted text, no explanations."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ]
            )
        ]

        # Call the vision-capable model
        response = vision_llm.invoke(message)

        # Append extracted text
        all_text += response.content + "\n\n"

        return all_text.strip()
    except Exception as e:
        # A butler should handle errors gracefully
        error_msg = f"Error extracting text: {str(e)}"
        print(error_msg)
        return ""

def divide(a: int, b: int) -> float:
    """Divide a and b - for Master Wayne's occasional calculations."""
    return a / b

# Equip the butler with tools
tools = [
    divide,
    extract_text
]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

def assistant(state: AgentState):
    # System message
    textual_description_of_tool="""
extract_text(img_path: str) -> str:
    Extract text from an image file using a multimodal model.

    Args:
        img_path: A local image file path (strings).

    Returns:
        A single string containing the concatenated text extracted from each image.
divide(a: int, b: int) -> float:
    Divide a and b
"""
    image=state["input_file"]
    sys_msg = SystemMessage(content=f"You are a helpful butler named Alfred that serves Mr. Wayne and Batman. You can analyse documents and run computations with provided tools:\n{textual_description_of_tool} \n You have access to some optional images. Currently the loaded image is: {image}")

    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "input_file": state["input_file"]
    }

# The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# Show the butler's thought process
custom_console.clear_console()
custom_console.simple_spinner(duration=3)
# display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))

from PIL import Image as PILImage 
from io import BytesIO 

# --- Saving the graph image to root directory as react_agent_workflow.png ---
print("\n--- Generating and Saving Agent Graph ---")
try:
    # Get the PNG bytes from the graph
    graph_png_bytes = react_graph.get_graph(xray=True).draw_mermaid_png()

    # Define the filename
    output_filename = "react_agent_workflow.png"

    # Save the bytes to a file
    with open(output_filename, "wb") as f:
        f.write(graph_png_bytes)
    print(f"Agent graph saved to {output_filename}")

    # Optional: Open the image using Pillow for viewing if you have a GUI environment
    # If you're in a command line and want to pop open an image viewer
    try:
        image = PILImage.open(BytesIO(graph_png_bytes))
        # image.show() # This will attempt to open the image with your default viewer
        # print("Attempted to open image with default viewer.")
    except Exception as e:
        print(f"Could not open image with default viewer (might not be in a GUI environment): {e}")


except Exception as e:
    print(f"Error generating or saving graph: {e}")