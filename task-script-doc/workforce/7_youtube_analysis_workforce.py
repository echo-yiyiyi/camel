from camel.models.qwen_model import QwenModel
from camel.toolkits.video_analysis_toolkit import VideoAnalysisToolkit
from camel.societies.workforce import Workforce
from camel.tasks.task import Task
from camel.agents.chat_agent import ChatAgent
from camel.messages.base import BaseMessage


def main():
    # Create Qwen model
    qwen_model = QwenModel(model_type="qwen-7b")

    # Create system message for ChatAgent
    sys_msg_qwen = BaseMessage.make_system_message(content="Qwen model worker")

    # Wrap Qwen model in ChatAgent with system message
    qwen_worker = ChatAgent(model=qwen_model, system_message=sys_msg_qwen)

    # Create video analysis toolkit
    video_analysis_toolkit = VideoAnalysisToolkit()

    # Create system message for video analysis worker
    sys_msg_video = BaseMessage.make_system_message(content="Video analysis worker")

    # Wrap video analysis toolkit in ChatAgent
    video_analysis_worker = ChatAgent(model=video_analysis_toolkit, system_message=sys_msg_video)

    # Create workforce
    workforce = Workforce('youtube_analysis_workforce')

    # Add Qwen model worker as a single agent worker
    workforce.add_single_agent_worker('qwen_worker', qwen_worker)

    # Add video analysis worker as a single agent worker
    workforce.add_single_agent_worker('video_analysis_worker', video_analysis_worker)

    # Create task to analyze the video
    task_content = "Analyze the video ../oasis_introduction.mp4 and provide insights."
    task = Task(content=task_content, id='youtube_analysis_1')

    # Process the task
    workforce.process_task(task)

    # Print workforce log tree
    print("\n--- Workforce Log Tree ---")
    print(workforce.get_workforce_log_tree())


if __name__ == '__main__':
    main()
