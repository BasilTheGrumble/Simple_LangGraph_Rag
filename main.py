from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages.tool import ToolMessage
from Embeddings_manipulations import search_similar_questions
import os


DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")


llm = ChatOpenAI(
    base_url="https://api.deepseek.com/beta",
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    temperature=0,
)


sys_msg = SystemMessage(content="""You are a helpful assistant tasked with searching for similar answers to answer a 
                                   user query using a set of inputs.""")
sys_msg_got_result = SystemMessage(content="""Do not use the tool, it has already returned the best answer from the 
                                              database (text). Your task is to evaluate this text. If the text is a good 
                                              answer to the question, then write only the answer to the question based 
                                              on this text. If the text is not suitable for answering the question, 
                                              output: 'I cannot answer this question; the answer is not in my knowledge 
                                              base.'""")

query1 = "I rented a car, and it broke down. What should I do?"
query2 = "Where can I buy a chandelier?"
query = query1

tools = [search_similar_questions]
llm_with_tools = llm.bind_tools(tools)


def assistant(state: MessagesState):
    if isinstance(state['messages'][-1], ToolMessage):
        return {
            "messages":
                [llm_with_tools.invoke(
                    [sys_msg_got_result] +
                    ['Вопрос: ', state["messages"][0].content] +
                    ['Текст из базы: ', state["messages"][-1].content]
                )
                ]
        }

    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine the control flow
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
    {
        "tools": "tools",
        END: END,
    }
)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
graph = builder.compile(checkpointer=MemorySaver())

initial_input = {"messages": HumanMessage(content=query)}

# Thread
config = {"configurable": {"thread_id": "1"}}

result = graph.invoke(initial_input, config)

latest_state = graph.get_state(config)

messages = latest_state.values.get('messages', [])

if messages:
    first_message = messages[0]
    last_message = messages[-1]
    first_message.pretty_print()
    print()
    last_message.pretty_print()
else:
    print("There are no any messages!")