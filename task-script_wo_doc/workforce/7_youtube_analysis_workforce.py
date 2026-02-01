from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.configs import QwenConfig
from camel.toolkits.video_analysis_toolkit import VideoAnalysisToolkit
from camel.societies.workforce.workforce import Workforce
from camel.societies.workforce.worker import Worker
from camel.types import ModelPlatformType, ModelType

class QwenModelWorker(Worker):
    def __init__(self):
        model = ModelFactory.create(
            model_platform=ModelPlatformType.DEFAULT,
            model_type=ModelType.QWEN_MAX,
            model_config_dict=QwenConfig().as_dict(),
        )
        super().__init__(description="Qwen model worker")
        self.name = "QwenModelWorker"
        self.model = model

    def _process_task(self, task_input):
        response = self.model.chat(task_input)
        return response

class VideoAnalysisWorker(Worker):
    def __init__(self):
        self.video_toolkit = VideoAnalysisToolkit()
        super().__init__(description="Video analysis worker")
        self.name = "VideoAnalysisWorker"

    def _process_task(self, video_path):
        question = "Please analyze the content of this video and provide a summary."
        result = self.video_toolkit.ask_question_about_video(video_path, question)
        return result

def create_youtube_analysis_workforce(video_path: str):
    workforce = Workforce(description="YouTube analysis workforce")

    qwen_worker = QwenModelWorker()
    video_worker = VideoAnalysisWorker()

    workforce.add_workforce(qwen_worker)
    workforce.add_workforce(video_worker)

    video_analysis_result = video_worker._process_task(video_path)
    qwen_summary = qwen_worker._process_task(video_analysis_result)

    return workforce, video_analysis_result, qwen_summary

if __name__ == "__main__":
    video_file_path = "../oasis_introduction.mp4"
    workforce, video_result, qwen_summary = create_youtube_analysis_workforce(video_file_path)

    print("Video Analysis Result:")
    print(video_result)
    print("\nQwen Model Summary of Video Analysis:")
    print(qwen_summary)
