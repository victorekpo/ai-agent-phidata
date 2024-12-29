## What is an AI Agent?
An AI agent is a system that perceives its environment and takes actions to achieve its goals. It can be a simple program that plays chess or a complex system that controls a self-driving car. AI agents can be classified into different types based on their capabilities and design. Here are some common types of AI agents:
1. Simple Reflex Agents: These agents take actions based on the current percept without considering the history of percepts. They are reactive and do not have memory or internal state. An example is a thermostat that turns on the heater when the temperature drops below a certain threshold.
2. Model-Based Reflex Agents: These agents maintain an internal model of the environment and use it to make decisions. They can consider the history of percepts and predict the future state of the environment. An example is a chess-playing program that simulates possible moves to choose the best one.
3. Goal-Based Agents: These agents have goals or objectives that guide their actions. They evaluate the current state of the environment and choose actions that maximize the likelihood of achieving their goals. An example is a robot vacuum cleaner that navigates a room to clean it efficiently.
4. Utility-Based Agents: These agents evaluate the desirability of different outcomes and choose actions that maximize their utility or satisfaction. They consider not only the likelihood of achieving a goal but also the value of the goal. An example is a financial trading system that selects investments to maximize profit.
5. Learning Agents: These agents improve their performance over time by learning from experience. They can adapt to new environments, tasks, or goals without explicit programming. An example is a self-driving car that learns to navigate roads by observing human drivers.
6. Rational Agents: These agents make decisions that maximize their expected utility given their knowledge and goals. They are idealized models of intelligent behavior and can be used to analyze the performance of other agents. An example is an AI agent that plays poker optimally against human opponents.
7. Multi-Agent Systems: These systems consist of multiple agents that interact with each other to achieve common or conflicting goals. They can be cooperative, competitive, or a mix of both. An example is a team of robots that collaborate to explore an unknown environment.
8. Autonomous Agents: These agents operate independently of human control and can make decisions without human intervention. They can be deployed in real-world applications such as autonomous vehicles, drones, or industrial robots.
9. Embodied Agents: These agents are situated in a physical or virtual environment and interact with it through sensors and actuators. They can perceive the environment, plan actions, and execute
10. Social Agents: These agents interact with humans or other agents in a social context. They can understand human emotions, intentions, and social norms to communicate effectively. An example is a chatbot that provides customer support or a virtual assistant that schedules meetings.
11. Intelligent Agents: These agents exhibit intelligent behavior by perceiving their environment, reasoning about it, and taking actions to achieve their goals. They can solve complex problems, adapt to changing conditions, and learn from experience. An example is an AI agent that plays a video game at a human level of performance.
12. Adaptive Agents: These agents can adapt to changing environments, tasks, or goals by adjusting their behavior or strategies. They can learn from feedback, optimize their performance, and evolve over time. An example is a recommendation system that personalizes content based on user preferences.
13. Cognitive Agents: These agents have cognitive capabilities such as perception, reasoning, planning, and learning. They can simulate human-like intelligence by processing information, making decisions, and adapting to new situations. An example is a virtual assistant that understands natural language commands and performs tasks on behalf of the user.
14. Robotic Agents: These agents are embodied in physical robots that interact with the environment through sensors and actuators. They can move, manipulate objects, and perform tasks in the physical world. An example is a humanoid robot that walks, talks, and interacts with humans in a social setting.

## Phidata Framework
What is Phidata?
Phidata is a framework for building multi-modal agents and workflows.

Build agents with memory, knowledge, tools and reasoning.
Build teams of agents that can work together to solve problems.
Interact with your agents and workflows using a beautiful Agent UI.
https://docs.phidata.com/agents/introduction

## To install
`pip3 install phidata groq openai duckduckgo-search newspaper4k lxml_html_clean`

Simple & Elegant
Phidata Agents are simple and elegant, resulting in minimal, beautiful code.

For example, you can create a web search agent in 10 lines of code.
```
# web_search.py

from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo

web_agent = Agent(
    name="Web Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)
web_agent.print_response("Tell me about OpenAI Sora?", stream=True)
```
python3 -m venv ~/.venvs/aienv
source ~/.venvs/aienv/bin/activate
pip install -U phidata openai duckduckgo-search
export OPENAI_API_KEY=sk-***
python web_search.py

### Multi-Modal by default
Phidata agents support text, images, audio and video.
For example, you can create an image agent that can understand images and make tool calls as needed
```
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo()],
    markdown=True,
)

agent.print_response(
    "Tell me about this image and give me the latest news about it.",
    images=["https://upload.wikimedia.org/wikipedia/commons/b/bf/Krakow_-_Kosciol_Mariacki.jpg"],
    stream=True,
)
```

### Multi-Agent orchestration
Phidata agents can work together as a team to achieve complex tasks.
```
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

agent_team.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=True)
```

Phidata provides a beautiful Agent UI for interacting with your agents.

https://docs.phidata.com/tools/toolkits

Send email with gmail
https://docs.phidata.com/tools/email

Read files from the disk
https://docs.phidata.com/tools/file

Pandas
https://docs.phidata.com/tools/pandas

Python - PythonTools enable an Agent to write and run python code.
https://docs.phidata.com/tools/python

Postgres - PostgresTools enable an Agent to interact with a PostgreSQL database.
https://docs.phidata.com/tools/postgres

Slack - 
https://docs.phidata.com/tools/slack

Shell - ShellTools enable an Agent to interact with the shell to run commands.
https://docs.phidata.com/tools/shell

