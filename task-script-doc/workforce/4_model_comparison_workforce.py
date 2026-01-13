"""
Script to create a workforce with workers using OpenAI, Gemini, and Qwen models to discuss which model performs best.
"""

from camel.agents.chat_agent import ChatAgent
from camel.models import ModelFactory
from camel.societies.workforce import Workforce
from camel.tasks.task import Task
from camel.messages.base import BaseMessage
from camel.types import ModelType, ModelPlatformType


def main():
    # Create system messages for each model worker
    openai_sysmsg = BaseMessage.make_assistant_message(
        role_name="OpenAI Worker",
        content="You are an AI worker using the OpenAI model. Discuss the performance of your model compared to others.",
    )
    gemini_sysmsg = BaseMessage.make_assistant_message(
        role_name="Gemini Worker",
        content="You are an AI worker using the Gemini model. Discuss the performance of your model compared to others.",
    )
    qwen_sysmsg = BaseMessage.make_assistant_message(
        role_name="Qwen Worker",
        content="You are an AI worker using the Qwen model. Discuss the performance of your model compared to others.",
    )

    # Create ChatAgent instances for each model
    openai_agent = ChatAgent(openai_sysmsg, model=ModelFactory.create(ModelPlatformType.DEFAULT, ModelType.GPT_4))
    gemini_agent = ChatAgent(gemini_sysmsg, model=ModelFactory.create(ModelPlatformType.DEFAULT, ModelType.GEMINI_2_5_PRO))
    qwen_agent = ChatAgent(qwen_sysmsg, model=ModelFactory.create(ModelPlatformType.DEFAULT, ModelType.QWEN_3_CODER_PLUS))

    # Create a workforce
    workforce = Workforce(description="Model Performance Comparison Workforce")

    # Add single agent workers for each model
    workforce.add_single_agent_worker(description="OpenAI Model Worker", worker=openai_agent)
    workforce.add_single_agent_worker(description="Gemini Model Worker", worker=gemini_agent)
    workforce.add_single_agent_worker(description="Qwen Model Worker", worker=qwen_agent)

    # Create a task for discussion
    discussion_task = Task(content="Discuss and compare the performance of OpenAI, Gemini, and Qwen models to determine which performs best.", id="model_comparison_task")

    # Process the task
    workforce.process_task(discussion_task)

    # Print workforce log tree
    print("\n--- Workforce Log Tree ---")
    print(workforce.get_workforce_log_tree())


if __name__ == "__main__":
    main()
