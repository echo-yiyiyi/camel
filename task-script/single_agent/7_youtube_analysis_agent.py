from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.toolkits.video_analysis_toolkit import VideoAnalysisToolkit
from camel.toolkits.video_download_toolkit import VideoDownloaderToolkit
from camel.types import ModelPlatformType, ModelType

# Create the main language model
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
)

# Create the video analysis toolkit with the model
video_analysis_toolkit = VideoAnalysisToolkit(model=model, use_audio_transcription=False)

# Create the video downloader toolkit
video_downloader_toolkit = VideoDownloaderToolkit()

# Combine tools from both toolkits
tools = [*video_analysis_toolkit.get_tools(), *video_downloader_toolkit.get_tools()]

# Create the agent with the combined tools
agent = ChatAgent(
    system_message="You are a helpful assistant that can download and analyze YouTube videos.",
    model=model,
    tools=tools,
)

# The YouTube video URL to download and analyze
# Changed to a different valid YouTube video URL for testing
video_url = "https://www.youtube.com/watch?v=BaW_jenozKc"  # This is a known test video URL

# The question to analyze the video
question = "Please analyze the content of the video and provide a summary."

# Use the video analysis toolkit to ask the question about the video
print("Downloading and analyzing video...")
result = video_analysis_toolkit.ask_question_about_video(video_url, question)

print("Analysis Result:")
print(result)
