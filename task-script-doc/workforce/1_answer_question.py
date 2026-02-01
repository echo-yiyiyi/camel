import asyncio
from camel.societies.workforce.workforce import Workforce
from camel.toolkits.search_toolkit import SearchToolkit
from camel.tasks.task import Task

async def main():
    question = ("What was the actual enrollment count of the clinical trial on H. pylori "
                "in acne vulgaris patients from Jan-May 2018 as listed on the NIH website?")

    # Create toolkits
    search_toolkit = SearchToolkit()

    # Create workforce
    workforce = Workforce("ClinicalTrialWorkforce")

    # Add role playing worker
    workforce.add_role_playing_worker(
        description="Search worker",
        assistant_role_name="assistant",
        user_role_name="user",
        assistant_agent_kwargs={"tools": [search_toolkit]},
        user_agent_kwargs=None,
        summarize_agent_kwargs=None,
    )

    # Create a task
    task = Task(content=question, id="1")

    # Process the task asynchronously
    result_task = await workforce.process_task_async(task)

    # Print the result
    print("Answer:", result_task.result)

if __name__ == "__main__":
    asyncio.run(main())
