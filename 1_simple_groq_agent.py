from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    markdown=True,
    instructions=["Use tables to display data."]
)

agent.print_response("Summarize and compare analyst recommendations and fundamentals for TSLA and NVDA listing last dates of price information that you have account of, let me know any thing even if its just the day, month, year that you have")