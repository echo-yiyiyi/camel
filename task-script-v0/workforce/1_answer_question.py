from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.societies.workforce import Workforce
from camel.tasks.task import Task

def main():
    """Create and run a workforce to answer a specific clinical trial question."""

    # Define the question task
    question = (
        "What was the actual enrollment count of the clinical trial on H. pylori in acne vulgaris patients "
        "from Jan-May 2018 as listed on the NIH website?"
    )

    # Create a workforce
    workforce = Workforce(description="Clinical Trial Data Analysis Team")

    # Add a search worker
    search_msg = BaseMessage.make_assistant_message(
        role_name="Search Worker",
        content="You are a worker specialized in searching clinical trial data and extracting relevant information.",
    )
    search_agent = ChatAgent(system_message=search_msg)
    workforce.add_single_agent_worker(description="Search Worker", worker=search_agent)

    # Add a thinking worker
    thinking_msg = BaseMessage.make_assistant_message(
        role_name="Thinking Worker",
        content="You are a worker specialized in analyzing and reasoning about clinical trial data.",
    )
    thinking_agent = ChatAgent(system_message=thinking_msg)
    workforce.add_single_agent_worker(description="Thinking Worker", worker=thinking_agent)

    # Add a code execution worker
    code_exec_msg = BaseMessage.make_assistant_message(
        role_name="Code Execution Worker",
        content="You are a worker specialized in executing code to process and analyze data.",
    )
    code_exec_agent = ChatAgent(system_message=code_exec_msg)
    workforce.add_single_agent_worker(description="Code Execution Worker", worker=code_exec_agent)

    # Create a Task object
    task = Task(content=question)

    # Run the workforce to process the task
    print("Running workforce to answer the clinical trial question...")
    result = workforce.process_task(task)

    # Print the result
    print("Answer:")
    print(result)


if __name__ == "__main__":
    main()
