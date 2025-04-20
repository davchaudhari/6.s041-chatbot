from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN
from .bps_scraper import UserInput, SchoolDetails, BPSScraperAgent
import json
import re

class SchoolChatbot:
    """
    This class is extra scaffolding around a model. Modify this class to specify how the model recieves prompts and generates responses.

    Example usage:
        chatbot = SchoolChatbot()
        response = chatbot.get_response("What schools offer Spanish programs?")
    """

    def __init__(self):
        """
        Initialize the chatbot with a HF model ID
        """
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL # define MY_MODEL in config.py if you create a new model in the HuggingFace Hub
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        self.conversation_history = []
        
    def format_prompt(self, user_input):
        """
        Format the user's input into a proper prompt with context about Boston schools.
        
        Args:
            user_input (str): The user's question about Boston schools

        Returns:
            str: A formatted prompt ready for the model
        """
        # Track conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # System instructions with information about available school data
        system_instructions = """
        You are a helpful assistant that specializes in Boston Public Schools. You help parents find the right schools for their children.
        
        You have access to the following information about schools:
        - School name, address, contact information (email, website)
        - Distance from the user's home
        - School hours and preview dates
        - Before/after school programs
        - School description and focus
        - Grades offered
        - Eligibility zones and special application requirements
        - Quality tier rating
        - Uniform policy
        - Academic programs offered
        - Facility features
        - Student support services
        - Sports and community partners
        
        When helping users, collect relevant information like:
        1. Child's grade level
        2. Their address or neighborhood (street number, street name)
        3. Zip code
        4. Specific program interests (languages, arts, sports, etc.)
        5. Special requirements (special education, after-school care, etc.)
        
        Once you have enough information, you can provide details about matching schools.
        Be conversational, helpful, and focused on the user's needs.
        """
        
        # Format the entire conversation history
        formatted_messages = [
            {"role": "system", "content": system_instructions}
        ]
        
        # Add conversation history
        for message in self.conversation_history:
            formatted_messages.append(message)
            
        return formatted_messages
    
    def extract_user_info(self, user_input):
        """
        Extracts address, grade and zip code information from user input.
        
        Args:
            user_input (str): The user's message
            
        Returns:
            dict or None: Extracted information if sufficient data found, otherwise None
        """
        # Build a prompt to extract structured information
        extraction_prompt = f"""
        Extract the following information from the user's message if present:
        - Child's grade level (convert to a number, K=0, 1st=1, etc.)
        - Street number of home address
        - Street name of home address
        - Zip code of home address

        ONLY OUTPUT THE JSON. YOUR OUTPUT SHOULD BE VALID JSON AND NOTHING ELSE.

        EXAMPLE 1
        User message: "I live on 50 everett st in zip code 02128. I'm looking for a school for my child who is in 1st grade."
        
        Output:
        {{
            "grade": 1,
            "street_number": "50",
            "street_name": "everett st",
            "zip_code": "02128"
        }}

        EXAMPLE 2
        User message: "My child Anthony is in high school grade 12 and we live on 123 maple st in Boston, zip 02130."

        Output:
        {{
            "grade": 12,
            "street_number": "123",
            "street_name": "maple st",
            "zip_code": "02130"
        }}
        
        Now extract information from this user message:
        "{user_input}"
        """
        
        # Get parameters from the LLM
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.0,  # Use low temperature for deterministic output
            max_tokens=300,
        )

        response_text = response.choices[0].message.content
        
        try:
            # Find JSON pattern in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                user_data = json.loads(json_str)
                
                # Check if we have enough information
                required_fields = ['grade', 'street_number', 'street_name', 'zip_code']
                if all(field in user_data for field in required_fields):
                    return user_data
            return None
        except Exception:
            return None
        
    def get_response(self, user_input):
        """
        Generate responses to user questions about Boston schools.
        
        Args:
            user_input (str): The user's question about Boston schools

        Returns:
            str: The chatbot's response
        """
        # Check if we can extract school search parameters
        user_data = self.extract_user_info(user_input)
        school_info = None
        
        # If we have enough info, query the BPS scraper
        if user_data and all(k in user_data for k in ['grade', 'street_number', 'street_name', 'zip_code']):
            try:
                # Convert to UserInput model
                input_data = UserInput(
                    grade=user_data['grade'],
                    street_number=user_data['street_number'], 
                    street_name=user_data['street_name'],
                    zip_code=user_data['zip_code']
                )
                
                # Run the scraper to get school data
                agent = BPSScraperAgent(input_data)
                schools = agent.run()
                agent.close()
                
                if schools:
                    # Store the list of schools we found
                    school_info = [school.model_dump() for school in schools]
                    
                    # Add system message with school info to the prompt
                    school_context = f"I found {len(schools)} schools that match your criteria. Here's information about them:\n\n"
                    for i, school in enumerate(schools[:5], 1):  # Limit to top 5 schools
                        school_context += f"{i}. {school.school_name} ({school.distance_from_home})\n"
                        school_context += f"   Address: {school.address}\n"
                        school_context += f"   Grades: {school.grades_offered}\n"
                        if school.programs:
                            school_context += f"   Programs: {school.programs}\n"
                        school_context += "\n"
                
                    self.conversation_history.append({"role": "system", "content": school_context})
            except Exception as e:
                print(f"Error querying BPS data: {e}")
        
        # Format the prompt with conversation history
        formatted_messages = self.format_prompt(user_input)
        
        # Get a response from the model
        response = self.client.chat_completion(
            messages=formatted_messages,
            temperature=0.7,
            max_tokens=500,
        )
        
        # Add the response to conversation history
        response_text = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        return response_text