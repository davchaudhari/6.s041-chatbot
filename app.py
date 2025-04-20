"""
Gradio Web Interface for Boston School Chatbot

This script creates a web interface for your chatbot using Gradio.
You only need to implement the chat function.

Key Features:
- Creates a web UI for your chatbot
- Handles conversation history
- Provides example questions
- Can be deployed to Hugging Face Spaces

Example Usage:
    # Run locally:
    python app.py
    
    # Access in browser:
    # http://localhost:7860
"""

import gradio as gr
from src.chat import SchoolChatbot

def create_chatbot():
    """
    Creates and configures the chatbot interface.
    """
    chatbot = SchoolChatbot()
    
    def chat(message, history):
        """
        Generate a response for the current message in a Gradio chat interface.
        
        This function is called by Gradio's ChatInterface every time a user sends a message.
        You only need to generate and return the assistant's response - Gradio handles the
        chat display and history management automatically.

        Args:
            message (str): The current message from the user
            history (list): List of previous message pairs, where each pair is
                           [user_message, assistant_message]

        Returns:
            str: The assistant's response to the current message.
        """
        # Generate response using our chatbot
        response = chatbot.get_response(message)
        return response

    
    
    # Create Gradio interface. Customize the interface however you'd like!
    demo = gr.ChatInterface(
        chat,
        title="Boston Public School Selection Assistant",
        description="Ask me anything about Boston public schools! I can help you find the right school for your child based on your address, zip code, and grade level.",
        examples=[
            "I live in Jamaica Plain and want to send my child to kindergarten. What schools are available?",
            "I live at 50 everett st in zip code 02128. I'm looking for a school for my 1st grader.",
            "What schools offer Spanish language programs?",
            "Are there schools with strong arts programs in Roxbury?",
            "My child needs special education services. What schools near 02115 would be good?"
        ]
    )
    
    return demo

if __name__ == "__main__":
    demo = create_chatbot()
    demo.launch()