import os
import base64
import custom_console
from smolagents import CodeAgent, ToolCollection, load_tool, ToolCallingAgent, DuckDuckGoSearchTool, FinalAnswerTool, models, Tool, tool, VisitWebpageTool
from dotenv import load_dotenv
import datetime
from langfuse import Langfuse
from mcp import StdioServerParameters

LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
LANGFUSE_AUTH=base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://us.cloud.langfuse.com/api/public/otel" # EU data region
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"
langfuse = Langfuse(
  secret_key=LANGFUSE_SECRET_KEY,
  public_key=LANGFUSE_PUBLIC_KEY,
  host=LANGFUSE_AUTH
)
from opentelemetry.sdk.trace import TracerProvider

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

# Loading UI in Console
custom_console.clear_console()
custom_console.simple_spinner(duration=3)
print('\n')

print(LANGFUSE_PUBLIC_KEY)
print(LANGFUSE_SECRET_KEY)
print(LANGFUSE_AUTH)
print(langfuse)


# 1. Load environment variables from the .env file
load_dotenv()

# 2. Get the API key value from the environment immediately
# and print it to confirm it's loaded by dotenv.
# Note - Only way to get the env variable to read.
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    # This check is vital. If this prints, your .env file isn't being read or key isn't there.
    print(f'{custom_console.COLOR_RED}//// Critical Error')
    print(f'{custom_console.RESET_COLOR}',"GOOGLE_API_KEY environment variable is NOT set.")
    print("Please ensure your .env file is in the same directory as this script and contains GOOGLE_API_KEY=YOUR_ACTUAL_KEY")
    exit(1) # Exit if the key is definitely not found

# This will show the first few characters of the key, confirming dotenv worked.
# print(f"DEBUG: GOOGLE_API_KEY detected in os.environ (first 5 chars): {google_api_key[:5]}")

# 3. Enable LiteLLM's internal debug mode
# litellm._turn_on_debug()
# print("DEBUG: LiteLLM debug mode is ON. Expect verbose output above the error.")

print("\nAttempting to run smolagents with Gemini model...")

try:
    @tool
    def suggest_menu(occasion: str) -> str:
        """
        Suggests a menu based on the occasion.
        Args:
            occasion (str): The type of occassion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Custom menu.
        """
        if occasion == "casual":
            return "Pizza, snacks, and drinks."
        elif occasion == "formal":
            return "3-course dinner with wine and dessert."
        elif occasion == "superhero":
            return "Buffet with high-energy and healthy food."
        else:
            return "Custom menu for the butler."

    @tool
    def catering_service_tool(query: str) -> str:
        """
        This tool returns the highest-rated catering service in Gotham City.
        
        Args:
            query: A search term for finding catering services.
        """
        # Example list of catering services and their ratings
        services = {
            "Gotham Catering Co.": 4.9,
            "Wayne Manor Catering": 4.8,
            "Gotham City Events": 4.7,
        }
    
        # Find the highest rated catering service (simulating search query filtering)
        best_service = max(services, key=services.get)
    
        return best_service

    class SuperheroPartyThemeTool(Tool):
        name = "superhero_party_theme_generator"
        description = """
        This tool suggests creative superhero-themed party ideas based on a category.
        It returns a unique party theme idea."""

        inputs = {
            "category": {
                "type": "string",
                "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham').",
            }
        }

        output_type = "string"

        def forward(self, category: str):
            themes = {
                "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
                "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
                "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
            }

            return themes.get(category.lower(), "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.")

    # 4. Initialize the smolagents Agent, explicitly passing the API key to LiteLLMModel
    # This bypasses LiteLLM's automatic environment variable lookup and forces the key.
    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool()],
        model=models.LiteLLMModel(
            model_id="gemini/gemini-1.5-flash",
            api_key=google_api_key
        ),
    )

    agent.run("Search for luxury superhero-themed party ideas, including decorations, entertainment, and catering.")


    # agent = CodeAgent(
    #     tools=[suggest_menu],
    #     model=models.LiteLLMModel(
    #         model_id="gemini/gemini-1.5-flash",
    #         api_key=google_api_key
    #     ),
    # )

    # agent.run("Prepate a Menu, but no one knows what they want")

    # agent = CodeAgent(
    #     tools=[],
    #     additional_authorized_imports=['datetime'],
    #     model=models.LiteLLMModel(
    #         model_id="gemini/gemini-1.5-flash",
    #         api_key=google_api_key
    #     ),
    # )

    # agent.run(
    # """
    # Alfred needs to prepare for the party. Here are the tasks:
    # 1. Prepare the drinks - 30 minutes
    # 2. Decorate the mansion - 60 minutes
    # 3. Set up the menu - 45 minutes
    # 4. Prepare the music and playlist - 45 minutes

    # If we start right now, at what time will the party be ready?
    # """
    # )

    # agent = CodeAgent(
    #     tools=[
    #         DuckDuckGoSearchTool(), 
    #         VisitWebpageTool(),
    #         suggest_menu,
    #         catering_service_tool,
    #         SuperheroPartyThemeTool(),
	#         FinalAnswerTool()
    #     ], 
    #     model=models.LiteLLMModel(
    #         model_id="gemini/gemini-1.5-flash",
    #         api_key=google_api_key
    #     ),
    #     max_steps=10,
    #     verbosity_level=2
    # )

    
    # agent.run("Give me the best playlist for a party at the Wayne's mansion. The party idea is a 'villain masquerade' theme")

    # agent = ToolCallingAgent(
    #     tools=[DuckDuckGoSearchTool()],
    #     model=models.LiteLLMModel(
    #         model_id="gemini/gemini-1.5-flash",
    #         api_key=google_api_key)
    # )

    # agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")

    # agent = CodeAgent(
    #     tools=[catering_service_tool], 
    #     model=models.LiteLLMModel(
    #         model_id="gemini/gemini-1.5-flash",
    #         api_key=google_api_key
    #     )
    # )

    # agent.run('Give me the highest rated catering service in Gotham.')

    # party_theme_tool = SuperheroPartyThemeTool()
    # agent = CodeAgent(
    #     tools=[party_theme_tool], 
    #     model=models.LiteLLMModel(
    #         model_id="gemini/gemini-1.5-flash",
    #         api_key=google_api_key
    #     )
    # )

    # result = agent.run(
    #     "What would be a good superhero party idea for a 'villain masquerade' theme?"
    # )

    # Need to subscribe to HuggingFace for more credits
    # image_generation_tool = load_tool(
    #     "m-ric/text-to-image",
    #     trust_remote_code=True
    # )
    # agent = CodeAgent(
    #     tools=[image_generation_tool],
    #     model=models.LiteLLMModel(
    #         model_id="gemini/gemini-1.5-flash",
    #         api_key=google_api_key
    #     )
    # )
    # agent.run("Generate an image of a luxurious superhero-themed party at Wayne Manor with made-up superheros.")

    # Using MCP Servers
    # server_parameters = StdioServerParameters(
    #     command="uvx",
    #     args=["--quiet", "pubmedmcp@0.1.3"],
    #     env={"UV_PYTHON": "3.12", **os.environ},
    # )
    
    # with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
    #     agent = CodeAgent(
    #         tools=[*tool_collection.tools], 
    #         model=models.LiteLLMModel(
    #             model_id="gemini/gemini-1.5-flash",
    #             api_key=google_api_key
    #         ),
    #         add_base_tools=True
    #     )
    #     agent.run("Please find a remedy for hangover.")

except Exception as e:
    print(f"\nCaught an error during agent.run: {type(e).__name__} - {e}")
    # The full LiteLLM error should have been printed above due to debug mode

