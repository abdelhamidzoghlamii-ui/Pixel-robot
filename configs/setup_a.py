# Setup A — Classic Stack
# Whisper (STT) + Qwen 3B (LLM) + YOLOv8m (Vision)

NAME = "setup_a_classic"
DESCRIPTION = "Whisper + Qwen 3B + YOLOv8m"

# LLM Server
LLM_MODEL = "/data/data/com.termux/files/home/models/qwen2.5-3b-instruct-q4_k_m.gguf"
LLM_PORT = 8080
LLM_CTX = 2048
LLM_CHAT_TEMPLATE = "qwen"  # <|im_start|> format

# STT
STT_ENGINE = "whisper"
WHISPER_BIN = "/data/data/com.termux/files/home/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/data/data/com.termux/files/home/whisper.cpp/models/ggml-base.bin"

# Vision
VISION_ENGINE = "yolo"
YOLO_MODEL = "/data/data/com.termux/files/home/robot/yolov8m.onnx"

# Prompt template for JSON actions
SYSTEM_PROMPT = (
    "You are a robot controller. Output ONLY a JSON array of actions. "
    "Available actions: navigate_to (field: room), find_person (field: name), say (field: message). "
    "Use minimum actions needed."
)

def build_prompt(user_input):
    return (
        "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
        "<|im_start|>user\n" + user_input + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
