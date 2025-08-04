from langchain import hub
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from app.services import LLM


@tool
def get_temperature() -> str:
    """Get the current temperature."""
    return "The current temperature is 20 degrees Celsius."


@tool
def get_date() -> str:
    """Get the current date."""
    return "Today's date is 2023-10-01."


if __name__ == "__main__":
    llm = LLM(provider="openai", model="gpt-4.1-2025-04-14")
    client = llm.get_client()
    tools = [get_temperature, get_date]

    try:
        prompt = hub.pull("hwchase17/react")

        agent = create_react_agent(client, tools, prompt)

        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        agent_executor.invoke({"input": "what is your name?"})

    except Exception as e:
        print(f"An error occurred: {e}")
        # print(
        #     f"Make sure to run ollama server, and make sure model is downloaded if using Ollama."
        # )
