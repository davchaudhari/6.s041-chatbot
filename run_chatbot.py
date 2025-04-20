from src.bps_chatbot import BPSChatbot
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    # Check if Hugging Face token is set
    if not os.getenv("HF_TOKEN"):
        print("Error: HF_TOKEN not found in environment variables")
        print("Please set your Hugging Face token in the .env file")
        return
        
    # Initialize the chatbot
    chatbot = BPSChatbot()
    chatbot.start()
    
    print("Welcome to the Boston Public Schools Enrollment Assistant!")
    print("I can help you find eligible schools for your child.")
    print("Please provide the following information:")
    print("1. Your child's name")
    print("2. Your home address")
    print("3. The grade your child is entering")
    print("\nType 'quit' to exit the conversation.")
    
    try:
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
                
            response = chatbot.process_message(user_input)
            print(f"\nAssistant: {response}")
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        chatbot.close()

if __name__ == "__main__":
    main() 