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
from camel.toolkits import BrowserToolkit
from camel.types import ModelPlatformType, ModelType


# Create a ChatAgent with browser tools and longterm memory enabled

def create_browser_agent_with_memory(role_name: str) -> ChatAgent:
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
    )

    web_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
    )

    planning_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
    )

    web_toolkit = BrowserToolkit(
        headless=True,
        web_agent_model=web_agent_model,
        planning_agent_model=planning_agent_model,
        channel="chromium",
    )

    system_message = f"You are {role_name}. You have longterm memory and browser tools to assist you."

    agent = ChatAgent(
        system_message=system_message,
        model=model,
        tools=[*web_toolkit.get_tools()],
    )

    return agent


async def main():
    workforce = Workforce("Browser Memory Workforce")

    # Add multiple agents with browser tools and longterm memory enabled
    for i in range(3):
        role = f"BrowserAgent_{i+1}"
        agent = create_browser_agent_with_memory(role)
        workforce.add_single_agent_worker(
            description=role,
            worker=agent,
            enable_workflow_memory=True,  # Enable longterm memory
        )

    # Create a task for the workforce to discuss
    task = Task(
        content="Discuss the impact of AI on modern society and how web browsing can help gather up-to-date information.",
        id="ai_impact_discussion",
    )

    # Process the task asynchronously
    await workforce.process_task_async(task)

    # Save the workflow memories after task completion
    saved_workflows = await workforce.save_workflow_memories_async()

    print("Saved workflow memories:", saved_workflows)


if __name__ == "__main__":
    asyncio.run(main())
