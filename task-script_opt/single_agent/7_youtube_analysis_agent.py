from camel.agents import ChatAgent
from camel.toolkits import VideoDownloaderToolkit, VideoAnalysisToolkit
from camel.models import ModelFactory
from camel.configs import ChatGPTConfig
from camel.types import ModelPlatformType, ModelType
from camel.messages import BaseMessage

# Create models for the agent and video analysis
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
)

video_model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
)

# Initialize toolkits
video_downloader_toolkit = VideoDownloaderToolkit()
video_analysis_toolkit = VideoAnalysisToolkit(
    model=video_model,
    use_audio_transcription=False,  # Disable audio transcription for faster processing
)

# Create an agent with both video downloader and video analysis tools
agent = ChatAgent(
    model=model,
    tools=[*video_downloader_toolkit.get_tools(), *video_analysis_toolkit.get_tools()],
)

# YouTube video URL to download and analyze
video_url = "https://www.youtube.com/shorts/SpoMkdAK4GU"

# Create a user message with the question
question = f"Download and analyze the YouTube video at this URL: {video_url}"
user_message = BaseMessage.make_user_message(role_name="User", content=question)

# Send the message to the agent and get the response
response = agent.step(user_message)

print("Agent response:")
print(response.msgs[0].content)
