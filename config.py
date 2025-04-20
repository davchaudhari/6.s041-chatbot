import os
from dotenv import load_dotenv

# Load from .env file. Store your HF token in the .env file.
load_dotenv()


BASE_MODEL ="HuggingFaceH4/zephyr-7b-beta"
# "meta-llama/Llama-2-7b-chat-hf"
# Other options:
# MODEL = "meta-llama/Llama-2-7b-chat-hf"
# MODEL = "openlm-research/open_llama_3b"

# If you finetune the model or change it in any way, save it to huggingface hub, then set MY_MODEL to your model ID. The model ID is in the format "your-username/your-model-name".
MY_MODEL = None
# MY_MODEL = "meta-llama/Llama-3.2-1B"

HF_TOKEN = os.getenv("HF_TOKEN")
