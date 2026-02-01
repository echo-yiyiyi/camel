from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.configs import QwenConfig, ChatGPTConfig
from camel.toolkits import VideoAnalysisToolkit
from camel.types import ModelPlatformType, ModelType
from camel.messages import BaseMessage

class Workforce:
    def __init__(self, workers):
        self.workers = workers

    def step(self, user_message):
        # Send the message to all workers and collect responses
        combined_content = []
        for worker in self.workers:
            # Wrap user_message in BaseMessage if it's a string
            if isinstance(user_message, str):
                msg = BaseMessage.make_user_message(role_name="User", content=user_message)
            else:
                msg = user_message
            response = worker.step(msg)
            combined_content.append(response.msgs[0].content)
        # Combine all responses into one message
        combined_response = "\n---\n".join(combined_content)
        # Return a dummy response object with combined content
        class DummyResponse:
            def __init__(self, content):
                self.msgs = [BaseMessage.make_assistant_message(role_name="Assistant", content=content)]
        return DummyResponse(combined_response)


# Create Qwen model worker
qwen_model = ModelFactory.create(
    model_platform=ModelPlatformType.QWEN,
    model_type=ModelType.QWEN_3_CODER_PLUS,
    model_config_dict=QwenConfig(temperature=0.2).as_dict(),
)
qwen_worker = ChatAgent(system_message="You are a helpful assistant.", model=qwen_model)

# Create video analysis worker
video_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
)
video_analysis_toolkit = VideoAnalysisToolkit(
    model=video_model,
    use_audio_transcription=False,
)
video_worker = ChatAgent(
    model=video_model,
    tools=video_analysis_toolkit.get_tools(),
)

# Create workforce with the two workers
workforce = Workforce(workers=[qwen_worker, video_worker])

# Define the video file path
video_file_path = "../oasis_introduction.mp4"

# Create a user message to analyze the video
user_message = f"Please analyze the video file located at: {video_file_path}"

# Send the task to the workforce
response = workforce.step(user_message)

# Print the response
print("Workforce response:")
print(response.msgs[0].content)
