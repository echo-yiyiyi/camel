# ========= Copyright 2023-2025 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2025 @ CAMEL-AI.org. All Rights Reserved. =========

from camel.agents import ChatAgent
from camel.memories import ScoreBasedContextCreator
from camel.memories.records import ContextRecord, MemoryRecord
from camel.messages import BaseMessage
from camel.types import RoleType, OpenAIBackendRole, ModelType
from camel.retrievers import HybridRetriever


def main():
    # Define the system message for the agent
    assistant_sys_msg = """You are a helpful assistant to answer questions.
    I will give you the Original Query and Retrieved Context,
    answer the Original Query based on the Retrieved Context,
    if you can't answer the question just say I don't know."""

    # Initialize the hybrid retriever to download and retrieve from Arxiv
    hybrid_retriever = HybridRetriever()
    # Use the Arxiv URL for the paper "Attention Is All You Need"
    arxiv_paper_url = "https://arxiv.org/abs/1706.03762"
    hybrid_retriever.process(content_input_path=arxiv_paper_url)

    # Query to answer
    query = "What is a Transformer?"

    # Retrieve relevant information
    retrieved_info = hybrid_retriever.query(
        query=query,
        top_k=5,
        vector_retriever_top_k=10,
        bm25_retriever_top_k=10,
    )

    # Prepare the user message with retrieved context
    user_msg = str(retrieved_info)

    # Create the chat agent
    agent = ChatAgent(assistant_sys_msg)

    # Get the assistant's response
    assistant_response = agent.step(user_msg)

    # Print the answer
    print("Question:", query)
    print("Answer:", assistant_response.msg.content)


if __name__ == "__main__":
    main()
