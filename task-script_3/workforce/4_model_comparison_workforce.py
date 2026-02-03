import asyncio
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types.enums import ModelPlatformType, ModelType
from camel.societies.workforce.workforce import Workforce
from camel.societies.workforce.single_agent_worker import SingleAgentWorker
from camel.tasks.task import Task

def main():
    # Create models
    openai_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI
    )

    gemini_model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type=ModelType.GEMINI_3_PRO
    )

    qwen_model = ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type=ModelType.QWEN_3_CODER_PLUS
    )

    # Create ChatAgents
    openai_agent = ChatAgent(model=openai_model)
    gemini_agent = ChatAgent(model=gemini_model)
    qwen_agent = ChatAgent(model=qwen_model)

    # Create workers with correct argument name 'worker'
    openai_worker = SingleAgentWorker(description="OpenAI_Worker", worker=openai_agent)
    gemini_worker = SingleAgentWorker(description="Gemini_Worker", worker=gemini_agent)
    qwen_worker = SingleAgentWorker(description="Qwen_Worker", worker=qwen_agent)

    # Create workforce with description only
    workforce = Workforce(description="Model Comparison Workforce")

    # Add workers to workforce
    workforce.add_single_agent_worker(description="OpenAI Worker", worker=openai_agent)
    workforce.add_single_agent_worker(description="Gemini Worker", worker=gemini_agent)
    workforce.add_single_agent_worker(description="Qwen Worker", worker=qwen_agent)

    # Define the task for discussion as a Task object
    task_content = "Discuss which model among OpenAI GPT-4o Mini, Gemini 3 Pro, and Qwen 3 Coder Plus performs best for general chat tasks."
    task = Task(content=task_content)

    # Run the workforce on the task asynchronously
    result = asyncio.run(workforce.process_task_async(task))

    # Print the result
    print("Discussion result:")
    print(result)


if __name__ == "__main__":
    main()
