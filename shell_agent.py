from prettytable import PrettyTable
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.shell import ShellTools
from dotenv import load_dotenv

load_dotenv()

# Initialize the agent with the model and tools
agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[ShellTools()],
    show_tool_calls=True,
    markdown=False  # Turn off markdown
)

def format_output_to_table(response):
    # Access the actual output from the response object
    output = response.content

    # Parse the output
    lines = output.splitlines()
    headers = ["File Name"]
    table = PrettyTable(headers)

    # Extract the file names from the numbered list
    for line in lines:
        if line.strip().startswith(tuple(str(i) for i in range(1, 10))):  # Check if the line starts with a number
            file_name = line.split('. ', 1)[1]  # Split by '. ' and get the second part
            table.add_row([file_name])

    # Return the formatted table
    return table

# Run the command using the agent
response = agent.run("Show me the contents of the current directory by listing the files, only run one command")
print("Response", response)
print(format_output_to_table(response))