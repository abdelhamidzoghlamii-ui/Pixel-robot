# Setup D — Hybrid
# Whisper (STT) + Qwen 3B (LLM) + Gemma 4 E2B (Vision)

NAME = "setup_d_hybrid"
DESCRIPTION = "Whisper + Qwen 3B (LLM) + Gemma 4 E2B (Vision)"

LLM_MODEL = "/data/data/com.termux/files/home/models/qwen2.5-3b-instruct-q4_k_m.gguf"
LLM_PORT = 8080
LLM_CTX = 2048
LLM_CHAT_TEMPLATE = "qwen"

VISION_MODEL = "/data/data/com.termux/files/home/models/gemma-4-e2b-it-q4_k_m.gguf"
VISION_PORT = 8081

STT_ENGINE = "whisper"
WHISPER_BIN = "/data/data/com.termux/files/home/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/data/data/com.termux/files/home/whisper.cpp/models/ggml-base.bin"

VISION_ENGINE = "gemma"

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

def build_vision_prompt(user_input, image_path):
    img_tag = "<img src=\'file://" + image_path + "\'>"
    return "<start_of_turn>user\n" + img_tag + "\n" + user_input + "<end_of_turn>\n<start_of_turn>model\n"
