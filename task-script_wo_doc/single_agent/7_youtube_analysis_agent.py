from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.toolkits.video_download_toolkit import VideoDownloaderToolkit
from camel.toolkits import VideoAnalysisToolkit
from camel.types import ModelPlatformType, ModelType

def create_youtube_analysis_agent():
    # Create the main chat model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(
            temperature=0.0,
        ).as_dict(),
    )

    # Create a model for video analysis
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

    # Initialize the VideoDownloaderToolkit
    video_downloader = VideoDownloaderToolkit(working_directory="downloads/")

    # Create an agent with both video downloader and video analysis tools
    agent = ChatAgent(
        system_message="You are a helpful assistant that can download and analyze YouTube videos.",
        model=model,
        tools=[*video_downloader.get_tools(), *video_toolkit.get_tools()],
    )

    return agent, video_downloader, video_toolkit

def main():
    video_url = "https://www.youtube.com/shorts/SpoMkdAK4GU"
    agent, video_downloader, video_toolkit = create_youtube_analysis_agent()

    print(f"Downloading video from {video_url} ...")
    video_path = video_downloader.download_video(video_url)
    if not video_path:
        print("Failed to download video. Please check the URL or try another video.")
        return
    print(f"Video downloaded to {video_path}")

    question = "Please analyze the content of this video and provide a summary."
    print("Analyzing video...")
    result = video_toolkit.ask_question_about_video(video_path, question)
    print("Analysis result:")
    print(result)

if __name__ == "__main__":
    main()
