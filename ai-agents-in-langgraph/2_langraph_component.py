from dotenv import load_dotenv
_ = load_dotenv()
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, system_prompt):
        self.system_prompt = system_prompt
        self.tools = { t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system_prompt: 
            messages = [SystemMessage(content=self.system_prompt)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}
    
    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0
    
    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:
                print("\n ...bad tool name...")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the models")
        return {'messages':results}

if __name__ == "__main__":
    tool = TavilySearchResults(max_results=4) #increased number of results
    print(type(tool))
    print(tool.name)

    prompt = """You are a smart research assistant. Use the search engine to look up information. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """
    model = ChatOpenAI(model="gpt-4o-mini")
    abot = Agent(model, [tool], system_prompt=prompt)

    question = """Who won the super bowl in 2024? In what state is the winning team headquarters located? \
    What is the GDP of that state? Answer each question."""

    messages = [HumanMessage(content=question)]
    result = abot.graph.invoke({"messages":messages})
    print(result['messages'][-1].content)