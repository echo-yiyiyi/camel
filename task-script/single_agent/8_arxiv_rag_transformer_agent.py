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
#!/usr/bin/env python3
import os

from dotenv import load_dotenv
from rich import print as rprint

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import ArxivToolkit
from camel.types import ModelPlatformType, ModelType

load_dotenv()


def main():
    rprint("[green]CAMEL AI with Arxiv Toolkit[/green]")

    # setup arxiv toolkit
    arxiv_toolkit = ArxivToolkit()
    tools = arxiv_toolkit.get_tools()
    rprint(f"Loaded [cyan]{len(tools)}[/cyan] tools")

    # setup gemini model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type=ModelType.GEMINI_2_5_PRO,
        api_key=os.getenv("GEMINI_API_KEY"),
        model_config_dict={"temperature": 0.5, "max_tokens": 40000},
    )

    # create agent with tools
    agent = ChatAgent(model=model, tools=tools)
    rprint("[green]Agent ready[/green]")

    # download the paper "Attention Is All You Need"
    download_response = agent.step("Download the paper 'Attention Is All You Need' from arXiv.")
    rprint(f"Download response: {download_response.msg}")

    # use vector retrieval to answer the question
    query = "What is a Transformer?"
    rprint(f"Query: {query}")
    response = agent.step(query)

    # show raw response
    rprint(f"\n[dim]{response.msg}[/dim]")

    # try to get the actual content
    if hasattr(response, 'msgs') and response.msgs:
        rprint(f"\nFound [cyan]{len(response.msgs)}[/cyan] messages:")
        for i, msg in enumerate(response.msgs):
            rprint(f"Message {i + 1}: {msg.content}")

    rprint("\n[green]Done[/green]")


if __name__ == "__main__":
    main()
