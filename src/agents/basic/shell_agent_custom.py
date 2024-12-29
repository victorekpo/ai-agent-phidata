from phi.agent import Agent
from dotenv import load_dotenv
from phi.model.groq import Groq

from src.tools.custom_shell_tool import ShellTools

load_dotenv()

# Initialize the agent with the model and tools
agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[ShellTools()],
    show_tool_calls=True,
    instructions=[
        "Always use ShellTools first before using the model.",
        "Use tables to display data.",
        "If ShellTools cannot handle the request, then use the model."
    ],
    markdown=False,  # Turn off markdown
    debug_mode=False
)

# Run the command using the agent
def run():
    agent.print_response("What is 2+2")

run()