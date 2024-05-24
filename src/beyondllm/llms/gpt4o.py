from beyondllm.llms.base import BaseLLMModel, ModelConfig
from typing import Any, Dict, List, Optional
import os
from dataclasses import dataclass, field
import base64
import subprocess, sys

@dataclass
class GPT4oOpenAIModel:
    """
    Class representing a Chat Language Model (LLM) model using OpenAI GPT-4o with Vision capabilities
    Example:
    from beyondllm.llms import GPT4OpenAIModel
    llm = GPT4OpenAIModel(model="gpt-4o", api_key = "", model_kwargs = {"max_tokens":512,"temperature":0.1})
    """

    api_key: str = ""
    model: str = field(default="gpt-4o")
    model_kwargs: Optional[Dict] = None

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not provided and not found in environment variables."
                )
        self.load_llm()

    def load_llm(self):
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI library is not installed. Please install it with ``pip install openai``.")

        try:
            self.client = openai.OpenAI(api_key=self.api_key)

        except Exception as e:
            raise Exception("Failed to load the model from OpenAI:", str(e))

        return self.client

    def _process_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _process_audio(self, audio_path):
        # Transcribe the audio
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1", file=open(audio_path, "rb")
        )
        return transcription.text

    def _process_video(self, video_path, seconds_per_frame=1):
        base64Frames = []
        base_video_path, _ = os.path.splitext(video_path)

        try:
            import cv2
        except ImportError:
            user_agree = input("The feature you're trying to use requires an additional library(s):opencv-python. Would you like to install it now? [y/N]: ")
            if user_agree.lower() == 'y':
                subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
                import cv2
            else:
                raise ImportError("The required 'opencv-python' is not installed.")

        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frames_to_skip = int(fps * seconds_per_frame)
        curr_frame = 0

        while curr_frame < total_frames - 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
            curr_frame += frames_to_skip
        video.release()

        audio_path = f"{base_video_path}.mp3"
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            user_agree = input("The feature you're trying to use requires an additional library(s):moviepy. Would you like to install it now? [y/N]: ")
            if user_agree.lower() == 'y':
                subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy"])
                from moviepy.editor import VideoFileClip
            else:
                raise ImportError("The required 'moviepy' is not installed.")
            
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, bitrate="32k")
        clip.audio.close()
        clip.close()

        return base64Frames, audio_path

    def predict(self, prompt: Any, media_paths: str = None):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # Ensure media_paths is a list if it's not None
        if media_paths:
            if isinstance(media_paths, str):
                media_paths = [media_paths]

            for media_path in media_paths:
                media_type = media_path.split(".")[-1].lower()
                if media_type in ["jpg", "png"]:
                    base64_image = self._process_image(media_path)
                    messages.append(
                        {
                            "role": "user",
                            "content": f"![image](data:image/{media_type};base64,{base64_image})"
                        }
                    )
                elif media_type in ["mp3", "wav"]:
                    transcription = self._process_audio(media_path)
                    messages.append(
                        {"role": "user", "content": f"The audio transcription is: {transcription}"}
                    )
                elif media_type in ["mp4", "avi", "webm"]:
                    base64Frames, audio_path = self._process_video(media_path)
                    transcription = self._process_audio(audio_path)
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                "These are the frames from the video.",
                                *map(
                                    lambda x: {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpg;base64,{x}",
                                            "detail": "low",
                                        },
                                    },
                                    base64Frames,
                                ),
                                {"type": "text", "text": f"The audio transcription is: {transcription}"},
                                {"type": "text", "text": prompt},
                            ],
                        },
                    )

                    # transcription = self._process_audio(audio_path)
                    # messages.append(
                    #     {"role": "user", "content": "These are the frames from the video:"}
                    # )
                    # for frame in base64Frames:
                    #     messages.append(
                    #         {
                    #             "role": "user",
                    #             "content": f"![frame](data:image/jpg;base64,{frame})"
                    #         }
                    #     )
                    # messages.append(
                    #     {"role": "user", "content": f"The audio transcription is: {transcription}"}
                    # )
                else:
                    raise ValueError(f"Unsupported media type: {media_type}")

        if self.model_kwargs is not None:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, **self.model_kwargs
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages
            )

        return response.choices[0].message.content

    @staticmethod
    def load_from_kwargs(self, kwargs):
        model_config = ModelConfig(**kwargs)
        self.config = model_config
        self.load_llm()