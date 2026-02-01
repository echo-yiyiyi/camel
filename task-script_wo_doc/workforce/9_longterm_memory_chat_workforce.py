from camel.societies.workforce.workforce import Workforce
from camel.memories.agent_memories import ChatHistoryMemory
from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.utils.token_counting import OpenAITokenCounter
from camel.types import ModelType
from camel.toolkits.browser_toolkit import BrowserToolkit
from camel.agents.chat_agent import ChatAgent


def main():
    # Define the task for the workforce to discuss
    task = "Discuss the impact of AI on long-term memory enhancement and browsing capabilities."

    # Extract keywords from the task (simple example)
    keywords = ["AI", "long-term memory", "browsing capabilities"]

    # Create a context creator and memory for workers
    context_creator = ScoreBasedContextCreator(token_counter=OpenAITokenCounter(ModelType.GPT_4O_MINI), token_limit=4096)
    memory = ChatHistoryMemory(context_creator=context_creator)

    # Create browser toolkit
    browser_toolkit = BrowserToolkit()

    # Create workers with memory and toolkits
    workers = []
    for i in range(3):
    worker = ChatAgent(
        memory=memory,
        toolkits_to_register_agent=[browser_toolkit],
    )
    worker.task = task
    workers.append(worker)
    workers[-1].task = task
    workers[-1].task = task
    workers[-1].task = task
    workers[-1].task = task
        worker = ChatAgent(
            memory=memory,
            toolkits_to_register_agent=[browser_toolkit],
            task=task
        )
        workers.append(worker)

    # Create workforce and assign workers
    workforce = Workforce(main_task=task)
    for worker in workers:
        workforce.pending_tasks.append(worker)

    # Run the workforce discussion
    workforce.run()

    # Print the keywords extracted
    print("Extracted keywords:", keywords)


if __name__ == "__main__":
    main()
