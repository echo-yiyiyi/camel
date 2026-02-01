"""
This script creates a workforce with multiple single-agent workers using different tools
(e.g., SearchToolkit) to collaboratively answer the question:
"What was the actual enrollment count of the clinical trial on H. pylori in acne vulgaris patients from Jan-May 2018 as listed on the NIH website?"
"""

from camel.agents.chat_agent import ChatAgent
from camel.messages.base import BaseMessage
from camel.models import ModelFactory
from camel.societies.workforce import Workforce
from camel.tasks.task import Task
from camel.toolkits import SearchToolkit
from camel.types import ModelPlatformType, ModelType


def main():
    # 1. Set up a Research agent with search tools
    search_agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Research Specialist",
            content="You are a research specialist who excels at finding and "
                    "gathering information from the web.",
        ),
        model=ModelFactory.create(
            model_platform=ModelPlatformType.DEFAULT,
            model_type=ModelType.DEFAULT,  # Use default model type
        ),
        tools=[SearchToolkit().search_duckduckgo],
    )

    # 2. Set up an Analyst agent
    analyst_agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Business Analyst",
            content="You are an expert business analyst. Your job is "
                    "to analyze research findings, identify key insights, "
                    "opportunities, and challenges.",
        ),
        model=ModelFactory.create(
            model_platform=ModelPlatformType.DEFAULT,
            model_type=ModelType.DEFAULT,
        ),
    )

    # 3. Set up a Writer agent
    writer_agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Report Writer",
            content="You are a professional report writer. You take "
                    "analytical insights and synthesize them into a clear, "
                    "concise, and well-structured final report.",
        ),
        model=ModelFactory.create(
            model_platform=ModelPlatformType.DEFAULT,
            model_type=ModelType.DEFAULT,
        ),
    )

    workforce = Workforce(
        'Clinical Trial Analysis Team',
        graceful_shutdown_timeout=30.0,
    )

    workforce.add_single_agent_worker(
        "A researcher who can search online for information.",
        worker=search_agent,
    ).add_single_agent_worker(
        "An analyst who can process research findings.", worker=analyst_agent
    ).add_single_agent_worker(
        "A writer who can create a final report from the analysis.",
        worker=writer_agent,
    )

    # specify the task to be solved
    human_task = Task(
        content=(
            "What was the actual enrollment count of the clinical trial on H. pylori in acne vulgaris patients "
            "from Jan-May 2018 as listed on the NIH website?"
        ),
        id='0',
    )

    workforce.process_task(human_task)

    # Print the workforce log tree and KPIs
    print("\n--- Workforce Log Tree ---")
    print(workforce.get_workforce_log_tree())

    print("\n--- Workforce KPIs ---")
    kpis = workforce.get_workforce_kpis()
    for key, value in kpis.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
