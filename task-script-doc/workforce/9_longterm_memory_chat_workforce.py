import asyncio
from camel.societies.workforce.workforce import Workforce
from camel.societies.workforce.worker import Worker
from camel.memories import LongtermAgentMemory
from camel.memories.base import BaseContextCreator
from camel.toolkits.browser_toolkit import BrowserToolkit
from camel.memories.records import ContextRecord
from camel.messages import OpenAIMessage
from typing import List, Tuple

class SimpleContextCreator(BaseContextCreator):
    @property
    def token_counter(self):
        return None  # Simplified for example

    @property
    def token_limit(self):
        return 1000  # Example token limit

    def create_context(self, records: List[ContextRecord]) -> Tuple[List[OpenAIMessage], int]:
        # Simplified context creation
        messages = [OpenAIMessage(role="system", content="This is a simple context.")]
        return messages, 0

class LongtermMemoryWorker(Worker):
    def __init__(self, name, toolkits):
        super().__init__(name)
        self.toolkits = toolkits
        self.memory = LongtermAgentMemory(context_creator=SimpleContextCreator())

    async def _process_task(self, task, dependencies):
        print(f"{self.name} is processing task: {task.content}")
        # Simulate using browser tools and memory
        print(f"{self.name} is using browser tools: {self.toolkits}")
        print(f"{self.name} has memory: {self.memory}")
        await asyncio.sleep(1)  # Simulate async task processing
        return 'DONE'


def create_longterm_memory_chat_workforce():
    workforce = Workforce(description="Longterm Memory Chat Workforce")
    task_content = "Discuss the impact of AI on society and future job markets."
    workforce.add_main_task(task_content)

    for i in range(5):
        worker = LongtermMemoryWorker(
            name=f"worker_{i+1}",
            toolkits=[BrowserToolkit()]
        )
        workforce.add_workforce(worker)

    return workforce

async def main():
    workforce = create_longterm_memory_chat_workforce()
    await workforce.start()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

