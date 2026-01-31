#!/usr/bin/env python3
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========

import asyncio

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.societies.workforce import Workforce
from camel.tasks.task import Task
from camel.toolkits.memory_toolkit import MemoryToolkit
from camel.types import ModelPlatformType, ModelType


def create_memory_agent() -> ChatAgent:
    """Create a memory agent with MemoryToolkit tools."""
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
        model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
    )

    agent = ChatAgent(
        system_message="You are a memory assistant that can save and recall information using memory tools.",
        model=model,
    )

    memory_toolkit = MemoryToolkit(agent=agent)
    for tool in memory_toolkit.get_tools():
        agent.add_tool(tool)

    return agent


def create_student_agent() -> ChatAgent:
    """Create a student agent to answer the butterfat content question."""
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
        model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
    )

    system_message = (
        "You are a student who answers questions based on given information. "
        "Answer the question precisely and concisely."
    )

    agent = ChatAgent(
        system_message=system_message,
        model=model,
    )

    return agent


async def main():
    workforce = Workforce("Memory Query Workforce")

    memory_agent = create_memory_agent()
    workforce.add_single_agent_worker(
        description="memory_worker",
        worker=memory_agent,
        enable_workflow_memory=True,
    )

    student_agent = create_student_agent()
    workforce.add_single_agent_worker(
        description="student_worker",
        worker=student_agent,
        enable_workflow_memory=False,
    )

    question = (
        "If this whole pint is made up of ice cream, how many percent above or below the US federal standards "
        "for butterfat content is it when using the standards as reported by Wikipedia in 2020? "
        "Answer as + or - a number rounded to one decimal place."
    )

    task = Task(content=question, id="butterfat_content_question")

    await workforce.process_task_async(task)


if __name__ == "__main__":
    asyncio.run(main())
