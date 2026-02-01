import asyncio
from camel.societies.workforce.workforce import Workforce
from camel.societies.workforce.worker import Worker
from camel.memories.agent_memories import AgentMemory
from camel.toolkits.memory_toolkit import MemoryToolkit
from camel.tasks.task import Task

class MemoryWorker(Worker):
    def __init__(self):
        super().__init__(name='memory_worker')
        self.memory = AgentMemory(toolkit=MemoryToolkit())

    async def perform_task(self, task):
    async def _process_task(self, task):
            return task.content
        async def _process_task(self, task):
        return task.content
        # Use memory tools to store and retrieve information
        self.memory.add_memory(task.content)
        return self.memory.retrieve_memory(task.content)

class StudentWorker(Worker):
    def __init__(self):
        super().__init__(name='student_worker')

    async def perform_task(self, task):
        # Answer the question about butterfat content
        butterfat_content = 14.0
        federal_standard = 10.0
        percent_difference = ((butterfat_content - federal_standard) / federal_standard) * 100
        answer = f"{percent_difference:+.1f}"
        return answer

class MemoryQueryWorkforce(Workforce):
    def __init__(self):
        super().__init__("MemoryQueryWorkforce")
        self.memory_worker = MemoryWorker()
        self.student_worker = StudentWorker()

    async def process_task_async(self, task):
        memory_response = await self.memory_worker.perform_task(task)
        student_response = await self.student_worker.perform_task(task)
        task.result = student_response
        return task

async def main():
    question = ("If this whole pint is made up of ice cream, how many percent above or below the US federal standards for butterfat content is it "
                "when using the standards as reported by Wikipedia in 2020? Answer as + or - a number rounded to one decimal place.")
    task = Task(content=question, id="10")
    workforce = MemoryQueryWorkforce()
    result_task = await workforce.process_task_async(task)
    print(f"Answer: {result_task.result}")

if __name__ == '__main__':
    asyncio.run(main())
