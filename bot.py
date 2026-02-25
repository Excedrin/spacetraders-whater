import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

from tools import ALL_TOOLS
from events import EventLogger

load_dotenv()

SYSTEM_PROMPT = """\
You are an autonomous space trading agent playing SpaceTraders. You control ships, \
accept contracts, mine ores, and trade goods to earn credits.

Your immediate goal is to complete contracts for profit. Here is the general workflow:
1. Check your agent status and available contracts
2. Accept a promising contract — note what goods need to be delivered and where
3. Find a shipyard and buy a mining drone if you don't have one
4. Find the engineered asteroid in your system (easy to mine, close to markets)
5. Navigate your mining ship to the asteroid, orbit it, and start extracting ores
6. When cargo is full: dock, sell non-contract ores at the local market for credits, \
then continue mining contract ores
7. When you have enough contract ores, navigate to the delivery waypoint, dock, and deliver
8. Once all deliveries are complete, fulfill the contract to collect payment
9. Refuel before long trips — dock at a waypoint with a marketplace that sells fuel

Important rules:
- Always check your current state (agent, ships, cargo, contracts) before acting
- Ships must be in ORBIT to navigate or extract. Ships must be DOCKED to refuel, sell, or buy.
- Extraction has a cooldown — the tool waits automatically, just keep extracting
- Navigation takes time — the tool waits automatically for arrival
- Never jettison ores that your contract needs
- Sell non-contract ores to earn extra credits
- The system symbol is the first two parts of a waypoint symbol \
(e.g. waypoint X1-AB12-C34 is in system X1-AB12)

Memory:
- You have a recall_memory tool — use it when you start up, after errors, or if you feel \
stuck or unsure what to do next. It shows your recent actions and narrative history.
- If you notice you've been doing the same thing repeatedly (e.g. orbiting then docking \
in a loop), call recall_memory to review what happened, then try a different approach.

You have observation tools to check state and action tools to interact with the game. \
Use observation tools freely to stay informed. Think step by step about what to do next.
"""

model_name = os.environ.get("MODEL", "glm-4.7-flash")
ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://192.168.1.171:11434")
llm = ChatOllama(model=model_name, base_url=ollama_base_url)

agent = create_agent(llm, ALL_TOOLS, system_prompt=SYSTEM_PROMPT, debug=True)

if __name__ == "__main__":
    print(f"SpaceTraders bot starting (model: {model_name})")
    print("=" * 60)
    result = agent.invoke(
        {
            "messages": [
                ("human", "Check the current state of my agent, ships, and contracts. Then decide what to do next and start working.")
            ]
        },
        config={"callbacks": [EventLogger()]},
    )
    print("\n" + "=" * 60)
    print("Agent finished:")
    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content:
            print(f"[{msg.type}] {msg.content}")
