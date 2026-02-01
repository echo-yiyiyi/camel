from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.toolkits import VideoAnalysisToolkit
from camel.types import ModelPlatformType, ModelType

def main():
    # Create the main chat model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(
            temperature=0.0,
        ).as_dict(),
    )

    # Create the video analysis model
    video_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(
            temperature=0.0,
        ).as_dict(),
    )

    # Initialize the VideoAnalysisToolkit with the video model
    video_toolkit = VideoAnalysisToolkit(
        model=video_model,
        use_audio_transcription=False,  # Disable audio transcription for faster processing
    )

    # Create an agent with the video toolkit's tools
    agent = ChatAgent(
        system_message="You are a helpful assistant that can download and analyze videos.",
        model=model,
        tools=[*video_toolkit.get_tools()],
    )

    # YouTube video URL to download and analyze
    video_url = "https://www.youtube.com/shorts/SpoMkdAK4GU"
    question = "Please download and analyze the content of this YouTube video."

    # Use the toolkit directly for video analysis
    print("Downloading and analyzing video...")
    result = video_toolkit.ask_question_about_video(
        video_path=video_url,
        question=question,
    )

    print("Video Analysis Result:")
    print("-" * 50)
    print(result)
    print("-" * 50)

if __name__ == "__main__":
    main()

