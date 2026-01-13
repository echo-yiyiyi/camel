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

"""
Workforce with Arxiv worker and vector memory worker to download "Attention Is All You Need" paper
and answer "What is a Transformer?".
"""

from camel.agents.chat_agent import ChatAgent
from camel.messages.base import BaseMessage
from camel.models import ModelFactory
from camel.societies.workforce import Workforce
from camel.societies.workforce.single_agent_worker import SingleAgentWorker
from camel.toolkits.arxiv_toolkit import ArxivToolkit
from camel.memories.agent_memories import ChatHistoryMemory
from camel.memories.blocks import VectorDBBlock
from camel.types import ModelPlatformType, ModelType

# Setup Arxiv worker agent
arxiv_agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="ArxivWorker",
        content="You are an expert researcher who downloads and summarizes academic papers from Arxiv."
    ),
    model=ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
    ),
)

# Setup Vector memory worker agent
vector_memory_agent = ChatAgent(
    system_message=BaseMessage.make_assistant_message(
        role_name="VectorMemoryWorker",
        content="You are a memory worker who uses vector memory to store and retrieve knowledge for answering questions."
    ),
    model=ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
    ),
)

# Initialize ArxivToolkit
arxiv_toolkit = ArxivToolkit()

# Create workforce
workforce = Workforce('Arxiv RAG Transformer Workforce')
workforce.add_single_agent_worker(
    "Arxiv paper downloader and summarizer",
    worker=SingleAgentWorker(arxiv_agent)
).add_single_agent_worker(
    "Vector memory knowledge retriever",
    worker=SingleAgentWorker(vector_memory_agent, memory=ChatHistoryMemory(VectorDBBlock()))
)

# Define the task
task = "Download the paper 'Attention Is All You Need' from Arxiv and answer the question: What is a Transformer?"

# Run the workforce on the task
result = workforce.run(task)

print("Task:", task)
print("Result:", result)
EOF && python3 task-script/workforce/8_arxiv_rag_transformer_workforce.py
