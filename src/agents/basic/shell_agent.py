from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.shell import ShellTools
from dotenv import load_dotenv

from src.utils.format_output import format_output_to_table

load_dotenv()

# Initialize the agent with the model and tools
agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[ShellTools()],
    show_tool_calls=True,
    markdown=False  # Turn off markdown
)

# Run the command using the agent
def run():
    response = agent.run("Show me the contents of the current directory by listing the files, only run one command")
    print("Response", response)
    print(format_output_to_table(response))