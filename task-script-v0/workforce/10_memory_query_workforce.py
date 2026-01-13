"""
Workforce with a memory worker and a student worker to answer a question about butterfat content.
"""

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.societies.workforce import Workforce
from camel.memories import ChatHistoryMemory
from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.toolkits.memory_toolkit import MemoryToolkit
from camel.models.model_factory import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.utils import OpenAITokenCounter

# Create a memory context creator with token counter
context_creator = ScoreBasedContextCreator(
    token_counter=OpenAITokenCounter(ModelType.GPT_4O_MINI),
    token_limit=1024
)

# Create a model for the agents
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
)

# Create a memory for the memory worker
memory = ChatHistoryMemory(context_creator=context_creator)

# Create the memory worker agent
memory_worker_system_msg = BaseMessage.make_system_message(
    content="You are a memory worker who manages and recalls relevant information from memory."
)
memory_worker = ChatAgent(
    system_message=memory_worker_system_msg,
    model=model,
    memory=memory,
    agent_id="memory_worker"
)

# Create the student worker agent
student_worker_system_msg = BaseMessage.make_system_message(
    content="You are a student worker who answers questions based on memory and knowledge."
)
student_worker = ChatAgent(
    system_message=student_worker_system_msg,
    model=model,
    agent_id="student_worker"
)

# Create the workforce
workforce = Workforce(description="Memory Query Workforce")

# Add the memory worker
workforce.add_single_agent_worker(description="Memory Worker", worker=memory_worker)

# Add the student worker
workforce.add_single_agent_worker(description="Student Worker", worker=student_worker)

# Define the task question
task_question = (
    "If this whole pint is made up of ice cream, how many percent above or below the US federal standards for butterfat content "
    "is it when using the standards as reported by Wikipedia in 2020? Answer as + or - a number rounded to one decimal place."
)

# Create a user message for the task
user_message = BaseMessage.make_user_message(content=task_question)

# Run the workforce to process the task
response = workforce.process_task(user_message)
print("Task Question:", task_question)
print("Response:", response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(workforce.process_task(user_message))
EOF && python3 task-script/workforce/10_memory_query_workforce.py
