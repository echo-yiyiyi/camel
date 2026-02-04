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
from camel.models import ModelFactory
from camel.types import ModelType, ModelPlatformType
from camel.toolkits import SearchToolkit


def main():
    # Create a SearchToolkit instance
    search_toolkit = SearchToolkit()

    # Get the search tools (including DuckDuckGo search) from the toolkit
    search_tools = search_toolkit.get_tools()

    # Create a Gemini model instance using ModelFactory
    gemini_model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,  # Gemini platform
        model_type=ModelType.GEMINI_2_5_FLASH  # Use a Gemini model type
    )

    # Create a ChatAgent with the Gemini model and the search tools
    agent = ChatAgent(
        "You are a helpful assistant.",
        model=gemini_model,
        tools=search_tools
    )

    # Example question to ask
    question = "What is the latest news about artificial intelligence?"

    # Get the agent's response
    response = agent.step(question)

    # Print the response content
    print("Agent response:")
    print(response.msgs[0].content)


if __name__ == "__main__":
    main()
