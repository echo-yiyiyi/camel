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

from camel.agents import ChatAgent
from camel.models.gemini_model import GeminiModel
from camel.toolkits.search_toolkit import SearchToolkit
from camel.configs import GeminiConfig
from camel.types import ModelType


def main():
    # Define system message
    system_message = "You are a helpful assistant."

    # Create Gemini model instance
    model = GeminiModel(
        model_type=ModelType.GEMINI_3_PRO,  # Use the gemini-3-pro model
        model_config_dict=GeminiConfig(temperature=0.2).as_dict(),
    )

    # Create SearchToolkit instance
    search_toolkit = SearchToolkit()

    # Get DuckDuckGo search tool from toolkit
    duckduckgo_tool = None
    for tool in search_toolkit.get_tools():
        if tool.func.__name__ == "search_duckduckgo":
            duckduckgo_tool = tool
            break

    tools = [duckduckgo_tool] if duckduckgo_tool else []

    # Create ChatAgent with system message, model, and tools
    agent = ChatAgent(system_message=system_message, model=model, tools=tools)

    # Example question to ask
    question = "Who is the current president of the United States?"

    # Get agent response
    response = agent.step(question)

    # Print the response content
    print(response.msgs[0].content)


if __name__ == "__main__":
    main()
