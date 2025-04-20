from chatbot_dev import SchoolChatbot
import time

def run_demo():
    # Initialize the chatbot
    print("Initializing Boston Public Schools Chatbot...")
    chatbot = SchoolChatbot()
    
    # Demo conversation flow
    demo_questions = [
        "Hi, I'm looking for a school for my child who will be in 1st grade. We live at 15 Hancock St, Boston MA 02114.",
        "Which schools have good after-school programs?",
        "What about schools with sports programs?",
        "Can you tell me more about the Eliot K-8 school?",
        "Thank you for your help!"
    ]
    
    print("\n=== Boston Public Schools Chatbot Demo ===\n")
    
    for question in demo_questions:
        print(f"User: {question}")
        print("\nChatbot is thinking...")
        
        # Get response from chatbot
        response = chatbot.get_response(question)
        
        print("\nChatbot:", response)
        print("\n" + "="*80 + "\n")
        
        # Add a small delay to make the demo more natural
        time.sleep(2)
    
    print("Demo completed. Thank you for using the Boston Public Schools Chatbot!")

if __name__ == "__main__":
    run_demo() 