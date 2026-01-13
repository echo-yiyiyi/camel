import asyncio
from camel.agents import ChatAgent
from camel.societies.workforce import Workforce
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.tasks.task import Task

async def main():
    # Create a workforce
    workforce = Workforce(description="Model Comparison Workforce")

    # Create workers with OpenAI, Gemini, and Qwen models
    openai_agent = ChatAgent(
        model=ModelFactory.create(ModelPlatformType.OPENAI, ModelType.GPT_4),
        system_message="You are an AI assistant using OpenAI GPT-4 model."
    )
    gemini_agent = ChatAgent(
        model=ModelFactory.create(ModelPlatformType.GEMINI, ModelType.GEMINI_2_5_PRO),
        system_message="You are an AI assistant using Gemini Pro model."
    )
    qwen_agent = ChatAgent(
        model=ModelFactory.create(ModelPlatformType.QWEN, ModelType.QWEN_MAX),
        system_message="You are an AI assistant using Qwen Max model."
    )

    # Add workers to the workforce
    workforce.add_single_agent_worker(description="OpenAI GPT-4 Worker", worker=openai_agent)
    workforce.add_single_agent_worker(description="Gemini Pro Worker", worker=gemini_agent)
    workforce.add_single_agent_worker(description="Qwen Max Worker", worker=qwen_agent)

    # Define the discussion task
    discussion_task = Task(content="Discuss and evaluate which model among OpenAI GPT-4, Gemini Pro, and Qwen Max performs best for general AI tasks.")

    # Assign the task to the workforce asynchronously
    results = await workforce.process_task_async(discussion_task)

    # Print the results
    print("Model Comparison Discussion Results:")
    for result in results:
        print(f"Worker: {result.worker_description}")
        print(f"Response: {result.response}")
        print("---")


if __name__ == '__main__':
    asyncio.run(main())
