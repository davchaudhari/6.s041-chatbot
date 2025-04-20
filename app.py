"""
Gradio Web Interface for Boston School Chatbot
----------------------------------------------

Run locally:
    python app.py
Then open the browser at http://localhost:7860

If you push this repo to Hugging Face Spaces with a
`requirements.txt` (gradio==4.* etc.), it will run there too.
"""

import gradio as gr
from src.chat import SchoolChatbot

def create_chatbot():
    """
    Builds and returns the Gradio ChatInterface
    for the Boston Public School selection assistant.
    """

    # One chatbot instance for the whole Space / local session
    bot = SchoolChatbot()

    def chat(message: str, history: list[list[str]]) -> str:
        """
        Generate a reply for the user's newest message.

        Args
        ----
        message : str
            The user's latest utterance.
        history : list of [str, str]
            Previous turns.  (Not needed by SchoolChatbot because it
            already tracks its own conversation state, but we can
            use the length of `history` to detect a fresh chat.)

        Returns
        -------
        str
            Assistant reply to show in the Gradio UI.
        """
        # Fresh chat panel?  Reset our internal state.
        if len(history) == 0:
            bot.reset_conversation()

        # Delegate to the domain‑specific chatbot
        return bot.get_response(message)

    def reset_conversation():
        """
        Callback for the ChatInterface “Clear” button.
        """
        bot.reset_conversation()
        # Returning None tells Gradio to clear the textbox
        return None

    # Build the UI.  You can tweak any aesthetics you like.
    demo = gr.ChatInterface(
        fn=chat,
        title="Boston Public School Selection Assistant",
        description=(
            "Ask anything about Boston Public Schools! "
            "The assistant will gather your child's grade level "
            "and address, search for eligible schools, and help you "
            "compare them. \n\n"
            "If you hit a 503 (free-tier model is busy), wait a few "
            "seconds and try again."
        ),
        examples=[
            "We live at 123 Maple St, 02130, and my daughter will be entering 1st grade.  What schools can we apply to?",
            "Show me schools near 50 Everett St 02128 with before school programs.",
            "Which elementary schools in Dorchester have Spanish immersion?"
        ]
    )

    return demo


if __name__ == "__main__":
    interface = create_chatbot()
    # For Spaces, you can omit server_name/port – HF sets them.
    interface.launch()
