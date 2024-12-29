import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from src.agents.basic.finance_agent import run as finance_agent
# from src.agents.basic.research_agent import run as research_agent
from src.agents.basic.shell_agent_gpt_4 import run as shell_agent

# finance_agent()
# research_agent()
shell_agent()
