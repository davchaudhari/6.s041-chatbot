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
            """
            This assistant helps you find schools for your child. To start, describe your child's exact (entering), grade level, exact street number, street name, and zip code. The assistant will then search for eligible schools near you and help you compare them. For instance, a parent might say: "My daughter is in 2nd grade, and we live on 95 Dunster St, Cambridge, MA 02138."
            """
        ),
        examples=[
            "We live at 46 Garden St, MA 02114, and my daughter will be entering 1st grade.",
            "My son will be a sophomore in high school, and we live at 124 Prince St, Boston, MA 02113",
            "My child will be entering eighth grade, and we live at 75 Cedar St, Boston, MA 02114"
        ],
        cache_examples=False
    )

    return demo


if __name__ == "__main__":
    interface = create_chatbot()
    # For Spaces, you can omit server_name/port – HF sets them.
    interface.launch()
