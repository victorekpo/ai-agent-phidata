from phi.agent import Agent
from phi.model.huggingface import HuggingFaceChat
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    model=HuggingFaceChat(
        id="yiyanghkust/finbert-tone",
        max_tokens=4096
    ),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    markdown=True,
    instructions=["Use tables to display data."]
)

def run():
    agent.print_response(
        "Summarize and compare analyst recommendations and fundamentals for TSLA and NVDA, list the current prices of the symbols also "
    )

run()