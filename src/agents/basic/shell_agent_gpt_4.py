from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.shell import ShellTools
from dotenv import load_dotenv

load_dotenv()

# Initialize the agent with the model and tools
agent = Agent(
    model=OpenAIChat(id="gpt-4"),
    tools=[ShellTools()],
    show_tool_calls=True,
    instructions=["Use tables to display data."],
    markdown=False  # Turn off markdown
)

# Run the command using the agent
def run():
    agent.print_response("Show me the contents of the current project by listing all files recursively, only run one command")