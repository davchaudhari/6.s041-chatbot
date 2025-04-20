import json
import os
from dotenv import load_dotenv
from src.bps_chatbot import BPSChatbot

load_dotenv()

EVAL_FILE = "eval_schema.json"

def run_eval_from_file(filepath):
    if not os.getenv("HF_TOKEN"):
        print("Error: HF_TOKEN not found in environment variables")
        return

    # Load eval schema
    with open(filepath, "r") as f:
        eval_data = json.load(f)

    convo = eval_data["conversation_flow_schema"]["conversation"]
    convo_id = eval_data["conversation_flow_schema"]["id"]
    print(f"\n📘 Running evaluation: {convo_id}")

    # Start chatbot
    chatbot = BPSChatbot()
    chatbot.start()

    # Walk through the conversation
    for turn in convo:
        role = turn["role"]
        content = turn["content"]

        if role == "user":
            print(f"\n👤 User: {content}")
            response = chatbot.process_message(content)
            print(f"🤖 Assistant: {response}")
        else:
            print(f"\n🤖 (Expected Assistant): {content}")
            if "expected_behavior" in turn:
                print(f"🧠 Expected Behavior: {json.dumps(turn['expected_behavior'], indent=2)}")

    chatbot.close()
    print("\n✅ Evaluation complete.\n")

def main():
    if not os.path.exists(EVAL_FILE):
        print(f"❌ Eval file not found: {EVAL_FILE}")
        return
    run_eval_from_file(EVAL_FILE)

if __name__ == "__main__":
    main()
