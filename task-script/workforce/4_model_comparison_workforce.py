from camel.agents.chat_agent import ChatAgent
from camel.societies.workforce.workforce import Workforce
from camel.models.model_factory import ModelFactory
from camel.types.enums import ModelPlatformType, ModelType


def main():
    # Create models for OpenAI, Gemini, and Qwen
    openai_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4,
    )

    gemini_model = ModelFactory.create(
        model_platform=ModelPlatformType.GOOGLE,
        model_type=ModelType.GEMINI_1_5,
    )

    qwen_model = ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type=ModelType.QWEN_2_5_7B,
    )

    # Create agents for each model
    openai_agent = ChatAgent("OpenAI Agent", model=openai_model)
    gemini_agent = ChatAgent("Gemini Agent", model=gemini_model)
    qwen_agent = ChatAgent("Qwen Agent", model=qwen_model)

    # Create a coordinator and task planner agent using OpenAI model
    coordinator_agent = ChatAgent("Coordinator Agent", model=openai_model)
    task_agent = ChatAgent("Task Planner Agent", model=openai_model)
    new_worker_agent = ChatAgent("New Worker Agent", model=openai_model)

    # Create the workforce
    workforce = Workforce(
        description="4 Model Comparison Workforce",
        coordinator_agent=coordinator_agent,
        task_agent=task_agent,
        new_worker_agent=new_worker_agent,
    )

    # Add workers for each model
    workforce.add_single_agent_worker(description="OpenAI Worker", worker=openai_agent)
    workforce.add_single_agent_worker(description="Gemini Worker", worker=gemini_agent)
    workforce.add_single_agent_worker(description="Qwen Worker", worker=qwen_agent)

    # Run a discussion task among the workers to compare models
    discussion_prompt = (
        "Discuss which model performs best among OpenAI, Gemini, and Qwen. "
        "Consider aspects like response quality, speed, and reliability."
    )

    # Use the workforce to run the discussion
    print("Starting model comparison discussion among workers...")
    results = workforce.run_discussion(discussion_prompt, max_turns=5)

    # Print the discussion results
    print("Discussion results:")
    for turn, message in enumerate(results, 1):
        print(f"Turn {turn}: {message}")


if __name__ == '__main__':
    main()
