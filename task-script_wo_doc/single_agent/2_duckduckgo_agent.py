from camel.agents import ChatAgent
from camel.embeddings.gemini_embedding import GeminiEmbedding
from camel.toolkits import SearchToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.toolkits.function_tool import FunctionTool
from camel.messages import BaseMessage
from camel.prompts import PromptTemplateGenerator
from camel.types import TaskType, RoleType

def create_duckduckgo_agent():
# Initialize Gemini embedding model
    gemini_embedding = GeminiEmbedding()

# Initialize SearchToolkit for DuckDuckGo search
    search_toolkit = SearchToolkit()
    search_duckduckgo_tool = FunctionTool(search_toolkit.search_duckduckgo)

# System message for the agent
    sys_msg = PromptTemplateGenerator().get_prompt_from_key(
TaskType.AI_SOCIETY, RoleType.ASSISTANT
)

# Create the chat agent with Gemini embedding model
    agent = ChatAgent(sys_msg)

# Add the search toolkit as a tool for the agent
    agent.add_tool(search_duckduckgo_tool)

    return agent

def main():
    agent = create_duckduckgo_agent()

# Example question to ask the agent
    question = "What is the latest news about artificial intelligence?"

# Create a user message with the question
    user_msg = BaseMessage.make_user_message(
role_name="User",
content=question,
)

# Get the agent's response
    response = agent.step(user_msg)

    print("Question:", question)
    print("Answer:", response.msg.content)


if __name__ == "__main__":
    main()
