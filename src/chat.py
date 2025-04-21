from huggingface_hub import InferenceClient
from .bps_scraper import UserInput, SchoolDetails, BPSScraperAgent
from config import BASE_MODEL, MY_MODEL, HF_TOKEN
from pydantic import ValidationError
import json
import re

class SchoolChatbot:
    """
    This class is extra scaffolding around a model. It handles conversations with parents
    looking for school information for their children.
    
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
        self.state = "information_collection"  # Initial state: collecting user information
        self.user_input_data = None  # Will store the UserInput object once collected
        self.school_list = None  # Will store the list of SchoolDetails once retrieved

        self._sent_school_summaries = False
        self.school_summaries = None # Will store formatted school summaries
        self._sent_school_context = False
        self.school_context = None  # Will store full formatted school information
    
    def extract_json(self, text: str) -> str:
        """
        Attempts to extract a valid JSON substring from the given text.
        First, it looks for a code block between triple backticks; if not found,
        it falls back to scanning between the first '{' and last '}'.
        """
        # Try to extract JSON within triple backticks (optionally labeled as json)
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        else:
            # Fallback: find first occurrence of '{' and last occurrence of '}'
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return text[start:end + 1]
            else:
                return ''
    
    def extract_query_parameters(self, user_input):
        """
        Extract parameters from user input that can be used to query the school database.
        
        Args:
            user_input (str): The user's question about Boston schools
            
        Returns:
            UserInput: Validated UserInput object with extracted parameters
        """
        # Build a prompt to extract structured information
        extraction_prompt = f"""
        Extract the following information from the user's message if present:
        - Child's grade level
        - Street number of home address
        - Street name of home address
        - Zip code of home address

        ONLY OUTPUT THE JSON. YOUR OUTPUT SHOULD BE VALID JSON AND NOTHING ELSE.

        EXAMPLE 1
        <|user|>
        I live on 50 everett st in zip code 02128. I'm looking for a school for my child who is 1st grade.
        
        <|assistant|>
        {{
            "grade": "1",
            "street_number": "50",
            "street_name": "everett st",
            "zip_code": "02128"
        }}

        EXAMPLE 2
        <|user|>
        My child Anthony is a junior in high school and we live on 123 maple st in zip code 02130. We're moving to Allston next month and need to find a school for him to attend.

        <|assistant|>
        {{
            "grade": "12",
            "street_number": "123",
            "street_name": "maple st",
            "zip_code": "02130"
        }}
        </END EXAMPLE 2>

        EXAMPLE 3
        <|user|>
        I'm looking for a good school in Jamaica Plain.

        <|assistant|>
        {{
            "grade": None,
            "street_number": None,
            "street_name": None,
            "zip_code": None
        }}
        </END EXAMPLE 3>

        EXAMPLE 4
        <|user|>
        I live on Elm Street in South Boston.

        <|assistant|>
        {{
            "grade": None,
            "street_number": None,
            "street_name": Elm Street,
            "zip_code": None
        }}
        </END EXAMPLE 4>

        EXAMPLE 5
        <|user|>
        My child is a freshman in high school and we live on 17 Milford St, Boston MA 02118.

        <|assistant|>
        {{
            "grade": "9",
            "street_number": "17",
            "street_name": "Milford Street",
            "zip_code": "02118"
        }}
        </END EXAMPLE 5>

        Consider all the conversation history to extract this information:
        {self.format_conversation_history()}
        
        Now include the current message:
        <|user|>
        {user_input}
        
        <|assistant|>
        """
        
        # Get parameters from the LLM
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.0,  # Use low temperature for deterministic output
            max_tokens=300,
        )

        response_text = response.choices[0].message.content
        
        # Parse the JSON response
        try:
            json_data = self.extract_json(response_text)
            user_input_obj = UserInput.model_validate_json(json_data)
            return user_input_obj
        except ValidationError as e:
            print("Validation error:", e.json())
            return None
    
    def query_school_database(self, user_input_obj):
        """
        Query the school database with the extracted user parameters
        
        Args:
            user_input_obj (UserInput): The validated UserInput object
            
        Returns:
            list: List of SchoolDetails objects
        """

        last_exception = None

        for attempt in range(1, 4):
            agent = BPSScraperAgent(user_input_obj)
            try:
                school_list = agent.run()
                print(f"Scraped {len(school_list)} schools on attempt {attempt}.")
                if len(school_list) > 0:
                    agent.close()
                    return school_list
                else:
                    print(f'No schools found on attempt {attempt}. Retrying...')

            except Exception as e:
                print(f"Warning: attempt {attempt} failed with error: {e}")
                last_exception = e
                try:
                    agent.close()
                except Exception:
                    pass  # ignore close errors
            
            print('Failed to scrape schools after 3 attempts.')
            raise last_exception

    def format_school_summaries(self, num_schools=10):
        """
        Format a list of school dictionaries into a structured, readable context
        that can be used effectively by a language model.
        
        Args:
            school_list (list): List of dictionaries containing school information
            
        Returns:
            str: A formatted string with school information in a clear, hierarchical structure
        """
        if not self.school_list:
            return "No schools information available."
        
        formatted_output = "# SCHOOLS INFORMATION\n\n"

        summary_stats = [el.model_dump(include={'school_name', 'distance_from_home', 'school_description', 'school_focus'}) for el in self.school_list]
        
        for i, school in enumerate(summary_stats[:num_schools], 1):
            # Main header with school name and distance
            formatted_output += f"## {i}. {school.get('school_name', 'Unnamed School')}\n"
            
            # Distance information if available
            if 'distance_from_home' in school:
                formatted_output += f"**Distance:** {school['distance_from_home']}\n\n"
            else:
                formatted_output += "\n"
            
            # School description
            if 'school_description' in school:
                formatted_output += "### Description\n"
                formatted_output += f"{school['school_description'].strip()}\n\n"
            
            # School focus
            if 'school_focus' in school:
                formatted_output += "### Focus\n"
                formatted_output += f"{school['school_focus'].strip()}\n\n"
            
            # Add any other fields that might be in the dictionary
            other_fields = [k for k in school.keys() if k not in ['school_name', 'distance_from_home', 'school_description', 'school_focus']]
            
            if other_fields:
                formatted_output += "### Additional Information\n"
                for field in other_fields:
                    # Convert field name from snake_case to Title Case for display
                    field_display = ' '.join(word.capitalize() for word in field.split('_'))
                    formatted_output += f"**{field_display}:** {school[field]}\n"
                formatted_output += "\n"
            
            # Add separator between schools
            if i < len(summary_stats[:num_schools]):
                formatted_output += "---\n\n"
        
        return formatted_output
            
    def format_school_list(self, num_schools=10):
        """
        Format a list of SchoolDetails objects into a readable, well-structured string
        that can be used as context for the language model.
        
        Args:
            school_list (list): List of SchoolDetails pydantic objects
            
        Returns:
            str: Formatted school information
        """
        if not self.school_list:
            return "No schools found matching your criteria."
        
        formatted_output = "# MATCHING SCHOOLS INFORMATION\n\n"
        
        for i, school in enumerate(self.school_list[:num_schools], 1):
            # Main header with school name and key info
            formatted_output += f"## {i}. {school.school_name}\n"
            formatted_output += f"**Distance:** {school.distance_from_home} | **Grades:** {school.grades_offered}\n\n"
            
            # Basic contact information
            formatted_output += "### Contact & Location\n"
            formatted_output += f"- **Address:** {school.address}\n"
            if school.school_website:
                formatted_output += f"- **Website:** {school.school_website}\n"
            if school.email:
                formatted_output += f"- **Email:** {school.email}\n"
            
            # Hours and scheduling
            if school.school_hours:
                formatted_output += f"- **Hours:** {school.school_hours}\n"
            formatted_output += "\n"
            
            # School profile
            formatted_output += "### School Profile\n"
            if school.school_description:
                formatted_output += f"- **Description:** {school.school_description.strip()}\n"
            if school.school_focus:
                formatted_output += f"- **Focus:** {school.school_focus.strip()}\n"
            if school.programs:
                formatted_output += f"- **Programs:** {school.programs}\n"
            if school.quality:
                formatted_output += f"- **Quality:** {school.quality}\n"
            formatted_output += "\n"
            
            # Admissions & eligibility
            formatted_output += "### Admissions Information\n"
            formatted_output += f"- **Eligibility:** {school.eligibility}\n"
            formatted_output += f"- **Special Application Required:** {school.special_application}\n"
            if school.preview_dates:
                formatted_output += f"- **Preview Dates:** {school.preview_dates}\n"
            if school.demand_reports:
                demand_info = school.demand_reports.replace("\n", " ").strip()
                formatted_output += f"- **Demand:** {demand_info}\n"
            formatted_output += "\n"
            
            # Additional services and facilities 
            formatted_output += "### Additional Features\n"
            if school.surround_care:
                formatted_output += f"- **Before/After School Programs:** {school.surround_care}\n"
            if school.facility_features:
                formatted_output += f"- **Facilities:** {school.facility_features}\n"
            if school.student_support:
                formatted_output += f"- **Student Support:** {school.student_support}\n"
            if school.sports:
                formatted_output += f"- **Sports:** {school.sports}\n"
            if school.community_partners:
                formatted_output += f"- **Community Partners:** {school.community_partners}\n"
            if school.uniform_policy:
                formatted_output += f"- **Uniform Policy:** {school.uniform_policy}\n"
            
            # Add separator between schools
            formatted_output += "\n" + "-" * 80 + "\n\n"
        
        return formatted_output
    
    def format_conversation_history(self):
        """
        Format the conversation history for model prompts.
        
        Returns:
            str: Formatted conversation history
        """
        if not self.conversation_history:
            return ""
            
        formatted_history = []
        for message in self.conversation_history:
            if message["role"] == "user":
                formatted_history.append(f"<|user|>\n{message['content']}")
            else:
                formatted_history.append(f"<|assistant|>\n{message['content']}")
                
        return "\n\n".join(formatted_history)
    
    # def generate__full_response(self, schools, user_input_obj):
    #     """
    #     Format the school results for the chatbot response.
        
    #     Args:
    #         schools (list): List of matching SchoolDetails objects
    #         user_input_obj (UserInput): The user input parameters
            
    #     Returns:
    #         str: Formatted school results
    #     """
    #     # Convert user_input_obj to a readable summary
    #     user_preferences = f"""
    #     - Grade level: {user_input_obj.grade_level}
    #     - Address: {user_input_obj.street_number} {user_input_obj.street_name}, {user_input_obj.zip_code}
    #     """
        
    #     # Build a prompt to format the results in a helpful way
    #     format_prompt = f"""
    #     <|system|>
    #     You have a list of schools that match the user's preferences.
        
    #     User preferences:
    #     {user_preferences}
        
    #     School information:
    #     {self.format_school_summaries(schools)}
        
    #     Provide a helpful, conversational response that summarizes the best options for the user.
    #     Focus on what's most relevant to the user's needs based on their grade level and location.
    #     Organize the information well and highlight key details that match their preferences.
    #     """
        
    #     # Get formatting from the LLM
    #     response = self.client.chat_completion(
    #         messages=[{"role": "user", "content": format_prompt}],
    #         temperature=0.7,
    #         max_tokens=2000,
    #     )
        
    #     return response.choices[0].message.content

    
    def handle_information_collection(self, user_input):
        """
        Handle the information collection phase of the conversation.
        
        Args:
            user_input (str): The user's question about Boston schools
            
        Returns:
            str: Response message
            bool: Whether to transition to query mode
        """
        # Check if we can extract the needed parameters
        extracted_params = self.extract_query_parameters(user_input)
        print('extracted params', extracted_params)

        if extracted_params is not None:
                
                # check if no fields in extracted_params are None

                if extracted_params.grade_level is not None and extracted_params.street_number is not None and extracted_params.street_name is not None and extracted_params.zip_code is not None:
                
                    # We have all the required information - collect schools
                    self.user_input_data = extracted_params
                    self.school_list = self.query_school_database(extracted_params)
                    
                    # Cache the formatted school information for future use

                    self.school_summaries = self.format_school_summaries()
                    self.school_list = self.format_school_list()
                    self._sent_school_summaries = True
                    self._sent_school_list = True

                    summarize_prompt = f"""
                    <|system|>
                    You are a helpful assistant that summarizes information about schools.
                    You have a list of schools that match the user's preferences.
                    You need to summarize the information about the schools in a way that is helpful to the user.

                    <EXAMPLE>
                    <|system|>
                    ## 1. New Mission High School
                    **Distance:** 10.170 mi from home

                    ### Description
                    New Mission is an exemplary college-prep school that develops high-achieving graduates who thrive in post-secondary education. Our dedicated faculty provide high-quality instruction. We offer programs including Homework Academy and Saturday Success Academy to provide support to students.

                    ### Focus
                    Early College and Career Pathways and Advisory Model. Pathways in computer science, engineering, health science, and business, via partnerships with UMass Boston and Wentworth Institute. Our Advisory Program combines students with a trusted adult at school together with strong family engagement.

                    ---

                    ## 3. Quincy Upper School
                    **Distance:** 1.650 mi from home

                    ### Description
                    JQUS offers the International Baccalaureate Program for ALL students grades 6-12, including the Arts and World Languages as core subjects, international travel opportunities, after-school programs and sports activities. Graduates matriculate into selective colleges, including Ivy League schools.

                    ### Focus
                    The mission of JQUS is to develop its students to be knowledgeable, productive members of a global society through an education that promotes cultural awareness, skillful use of information, personal renewal, and both individual and community path-finding.

                    ---

                    <|assistant|>
                    Here are some eligible schools based on your location and grade level:
                    1. New Mission High School (10.2 mi away): A top-tier college-prep school offering Early College and Career Pathways in fields like computer science and health science through UMass Boston and Wentworth partnerships; features strong academic support programs and a trusted adult Advisory Model for personalized guidance.

                    2. Charlestown High School (1.5 mi away): A diverse, community-driven school with strong support for English language learners and up to 30 college credits available through its rigorous Pathways Program; emphasizes family engagement and personalized career-focused education.

                    3. Quincy Upper School (1.7 mi away): Offers the rigorous International Baccalaureate (IB) Program to all students in grades 6-12, with a global focus that includes world languages, arts, and international travel; prepares students for selective colleges and emphasizes cultural awareness and global citizenship.
                    </EXAMPLE>

                    Previous conversation:
                    {self.format_conversation_history()}
                    
                    School information:
                    {self.school_summaries}
                    
                    <|assistant|>
                    """

                    response = self.client.chat_completion(
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that summarizes information about schools a child is eligible for."},
                            {"role": "user", "content": summarize_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1000,
                    )

                    response_text = response.choices[0].message.content
                    self.conversation_history.append({"role": "assistant", "content": response_text})
                    
                    # Transition to query mode
                    return response_text, True
                
                else:
                    # We need more information - ask the user

                    missing_info_dict = {
                        'grade_level': 'exact grade level (e.g. "1st grade", "8th grade", "12th grade")',
                        'street_number': 'street number (e.g. "123" or "291"). Not necessary to specify unit number',
                        'street_name': 'street name (e.g. "maple st" or "hendrix ave"). Not sufficient to just specify neighborhood',
                        'zip_code': 'zip code (e.g. "02130" or "02114")'
                    }

                    # Create list of missing fields
                    missing_fields = [field for field, value in extracted_params.dict().items() 
                                    if value is None]
                    
                    # Build prompt asking for missing information
                    missing_info_list = [missing_info_dict[field] for field in missing_fields]
                    missing_info_text = ", ".join(missing_info_list[:-1])
                    if len(missing_info_list) > 1:
                        missing_info_text += f", and {missing_info_list[-1]}"
                    else:
                        missing_info_text = missing_info_list[0]
                        
                    missing_info_prompt = f"""
                    <|system|>
                    You are helping a parent find schools for their child in Boston.
                    You need to collect:
                    1. The child's grade level
                    2. Their home address (exact street number, exact street name, and exact zip code)
                    
                    Based on what the user has told you so far: "{user_input}"
                    
                    Ask for any missing information in a friendly, conversational way.
                    Mention specifically what information you need to help find schools.
                    You need to be very concise and to the point.
                    
                    Previous conversation:
                    {self.format_conversation_history()}
                    
                    <|assistant|>
                    """
                    
                    response = self.client.chat_completion(
                        messages=[
                            {"role": "system", "content": "You are helping a parent find schools for their child in Boston."},
                            {"role": "user", "content": missing_info_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=300,
                    )
                    
                    response_text = response.choices[0].message.content
                    self.conversation_history.append({"role": "assistant", "content": response_text})
                    
                    # Stay in information collection mode
                    return response_text, False
                        
        else:
            # We need more information - ask the user
            missing_info_prompt = f"""
            <|system|>
            You are helping a parent find schools for their child in Boston.
            You need to collect:
            1. The child's grade level
            2. Their home address (exact street number, exact street name, and exact zip code)
            
            Based on what the user has told you so far: "{user_input}"
            
            Ask for any missing information in a friendly, conversational way.
            Mention specifically what information you need to help find schools.
            You need to be very concise and to the point.
            
            Previous conversation:
            {self.format_conversation_history()}
            
            <|assistant|>
            """
            
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are helping a parent find schools for their child in Boston."},
                    {"role": "user", "content": missing_info_prompt}
                ],
                temperature=0.7,
                max_tokens=300,
            )
            
            response_text = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Stay in information collection mode
            return response_text, False
    
    def handle_query_mode(self, user_input, num_schools=10):
        """
        Handle the query mode phase of the conversation (after schools list is retrieved).
        
        Args:
            user_input (str): The user's follow-up question
            
        Returns:
            str: Response message
        """
        # If school context is not cached yet, create it
        if not self.school_context and self.school_list:
            self.school_context = self.format_school_list()
            self._sent_school_context = True
        
        # Create a prompt to answer the specific question with the filtered schools
        answer_prompt = f"""
        <|system|>
        The user is asking for specific information about Boston schools.
        Their question is: "{user_input}"

        We've already generated a list of schools that are close to the user's home. For each school, we know the following:
        - School name, address, website, and email
        - Distance from home
         - Schoool hours, preview dates (for prospective students), and surround care (before/after school programs)
         - School description, focus, programs, quality, and facility features
        - Grades offered, special application procedures (for enrollment), and uniform policy (if any)
        - Student support, sports, and community partners

        We'll provide information about the five closest schools that match their criteria. Here is an example of the information we have, user queries, and how you should respond:

                <EXAMPLE 1>
        <|system|>
        ## 1. Mario Umana Academy
        **Distance:** 0.345 mi from home | **Grades:** K1 - 8

        ### Contact & Location
        - **Address:** 312 Border St East Boston MA 02128
        - **Website:** http://bostonpublicschools.org/Page/628
        - **Email:** umana@bostonpublicschools.org
        - **Hours:** 7:10am - 2:00pm

        ### School Profile
        - **Description:** We embrace and celebrate diverse languages and cultures. We are dedicated to a diverse and inclusive Umana community, where everyone can be themselves, and where we foster learning and support through collaboration with students, families, and the community.
        - **Focus:** The Mario Umana Academy is the only Dual Language school in East Boston, where students receive instruction in both English and Spanish. As an inclusive school, we serve students with autism and intellectual impairments along side their typically developing peers.
        - **Programs:** Arts, Dual Language, Phys Education
        - **Quality:** BPS Quality Tier - 2
        BPS Quality Report
        More information on school quality measures

        ### Admissions Information
        - **Eligibility:** 1 Mile:Closest Tier 2+:Closest Tier 3+:East Boston
        Distance: 0.285 mi
        - **Special Application Required:** No
        - **Preview Dates:** (P) In-Person Session; (V) - Virtual Session
        • 11/14/2024, 8:30 AM - 9:30 AM, (P)
        • 12/12/2024, 8:30 AM - 9:30 AM, (P)
        • 1/16/2025, 8:30 AM - 9:30 AM, (P)
        - **Demand:** See Seat and applicant data: Latest Demand Report Historic Demand Data

        ### Additional Features
        - **Before/After School Programs:** After: Umana After School Program is a free K-5th grade program, providing academic and enrichment components, until 4:30pm. We also partner with the YMCA to provide programming for students in K-5th grade until 6:00pm
        - **Facilities:** Art Room, Athletic Field, Auditorium, Cafeteria, Computer Lab, Gymnasium, Library, Music Room, Outdoor Classrooms, Playground, Pool, Science Lab
        - **Student Support:** Family Coordinator, Full-Time Nurse, Guidance Counselor, Social Worker
        - **Sports:** Basketball, Hockey, Soccer, Track
        - **Community Partners:** Tenacity, , Gear Up, Boston Debate League, E-Inc., Waypoint Adventure, America Scores Soccer, YMCA of East Boston, East Boston Neighborhood Health Center, Eastie Farms
        - **Uniform Policy:** Uniform tops: Yellow or Navy Blue; Uniform bottoms - Navy Blue or Khaki

        --------------------------------------------------------------------------------

        ## 2. O'Donnell Elementary School
        **Distance:** 0.350 mi from home | **Grades:** K1 - 6

        ### Contact & Location
        - **Address:** 33 Trenton St East Boston MA 02128
        - **Website:** http://bostonpublicschools.org/Page/628
        - **Email:** odonnell@boston.k12.ma.us
        - **Hours:** 9:30am - 4:10pm

        ### School Profile
        - **Description:** The O'Donnell is a small community school that our students will describe as their second home. At the O'Donnell, students receive access to a high-quality and rigorous education and are part of a community that values each child for the human they are and will become.
        - **Focus:** A majority of our students are multilingual or are becoming multilingual. We focus first on creating a safe and welcoming environment for all, and then on providing a high quality education while providing access for all of our learners, including those with language needs.
        - **Quality:** BPS Quality Tier - 1
        BPS Quality Report
        More information on school quality measures

        ### Admissions Information
        - **Eligibility:** 1 Mile:Closest Tier 1:Closest Tier 2+:Closest Tier 3+:East Boston
        Distance: 0.279 mi
        - **Special Application Required:** No
        - **Preview Dates:** (P) In-Person Session; (V) - Virtual Session
        • 12/4/2024, 3:00 PM - 4:00 PM, (P)
        • 12/18/2024, 10:00 AM - 11:00 AM, (P)
        • 1/23/2025, 10:00 AM - 11:00 AM, (P)
        - **Demand:** See Seat and applicant data: Latest Demand Report Historic Demand Data

        ### Additional Features
        - **Before/After School Programs:** Before: Scores, Strings Lessons. (We will create an on-site before/after school based on demand for 2025-2026)

        After: Scores, Strings Lessons. (We will create an on-site before/after school based on demand for 2025-2026)
        - **Facilities:** Cafeteria, Outdoor Classrooms, Playground
        - **Student Support:** Family Coordinator, Full-Time Nurse, Part-Time Nurse
        - **Sports:** Soccer
        - **Community Partners:** Boston Community Music Center, , Playworks, Boston University, Wallace Grant, Isabella Stewart Gardner Museum, Boston Scores Soccer Program, YMCA Before and After-School Program, ,
        - **Uniform Policy:** Navy tops, navy or khaki bottoms

        --------------------------------------------------------------------------------

        ## 3. Kennedy Patrick J Elementary School
        **Distance:** 0.602 mi from home | **Grades:** K0 - 6

        ### Contact & Location
        - **Address:** 343 Saratoga St East Boston MA 02128
        - **Website:** http://bostonpublicschools.org/Page/628
        - **Email:** pkennedy@bostonpublicschools.org
        - **Hours:** 8:30am - 3:10pm

        ### School Profile
        - **Description:** Patrick J. Kennedy Elementary, located in East Boston, is a small, close-knit school with about 300 students. Our dedicated staff of 40+ professionals provides a safe, caring, and stimulating environment where students grow academically and emotionally, fostering a family-like atmosphere.
        - **Focus:** Our Instructional Focus: Educators will provide explicit instruction in phonemic awareness, phonics, and word analysis, while engaging students with complex texts in Tier 1 instruction across content areas. We will document effective Tier 2 and Tier 3 interventions to strengthen our MTSS model.
        - **Programs:** Arts, Inclusion, Phys Education
        - **Quality:** BPS Quality Tier - 3
        BPS Quality Report
        More information on school quality measures

        ### Admissions Information
        - **Eligibility:** 1 Mile:Closest Tier 3+:East Boston
        Distance: 0.47 mi
        - **Special Application Required:** No
        - **Preview Dates:** (P) In-Person Session; (V) - Virtual Session
        • 11/22/2024, 2:00 PM - 2:45 PM, (V)
        • 12/9/2024, 2:00 PM - 2:45 PM, (P)
        • 1/17/2025, 9:00 AM - 9:45 AM, (V)
        - **Demand:** See Seat and applicant data: Latest Demand Report Historic Demand Data

        ### Additional Features
        - **Before/After School Programs:** Before: America SCORES Soccer Program for Gr. 3-6; Breakfast for all students begins at 8:00 am

        After: We run our own after school program daily from 3:10 to 5:30pm, which includes afterschool meal, homework help, sports, and activities offered by student choice. - Boston Scores Soccer - Girls on the Run - Figure Skating Club
        - **Facilities:** Athletic Field, Cafeteria, Music Room, Outdoor Classrooms, Playground, Science Lab
        - **Student Support:** Family Coordinator, Full-Time Nurse, Part-Time Nurse
        - **Community Partners:** America SCORES, Home for Little Wanderers, BOKS, Fresh Truck, CitiArts, Community Music Center, Shah Foundation, Emmanuel College, Boston College, UMASS TeachU, Mount Ida, FoodCorps, Kiwanis
        - **Uniform Policy:** PJK maroon shirt or sweatshirt with PJK crest (purchased on our PJK storefront website), khaki or black bottoms, or PJK maroon sweatpants

        <|user|>
        What are the preview dates for the schools?

        <|assistant|>
        Mario Umana Academy is offering in-person preview sessions on November 14, December 12, and January 16, all from 8:30 to 9:30 AM. O'Donnell Elementary School also has in-person sessions scheduled for December 4 from 3:00 to 4:00 PM, December 18 from 10:00 to 11:00 AM, and January 23 from 10:00 to 11:00 AM. Kennedy Patrick J Elementary School will host a mix of virtual and in-person sessions: a virtual session on November 22 from 2:00 to 2:45 PM, an in-person session on December 9 from 2:00 to 2:45 PM, and another virtual session on January 17 from 9:00 to 9:45 AM.

        <|user|>
        What are the main differences between the schools?

        <|assistant|>
        The main differences between the schools lie in their focus, language offerings, and quality ratings. Mario Umana Academy stands out as East Boston's only Dual Language school (English-Spanish) and serves a diverse, inclusive student body, including those with special needs (Tier 2 quality). O'Donnell Elementary is a small, community-oriented school with a strong emphasis on multilingual learners and holds the highest quality rating (Tier 1). Kennedy Patrick J Elementary offers a close-knit environment with a focus on foundational literacy skills and intervention support but has a lower quality rating (Tier 3). All three schools offer after-school programs and community partnerships, but vary in facilities and uniform policies.
        
        <|user|>
        What's the email of Kennedy Patrick J Elementary School?
        <|assistant|>
        The email for Kennedy Patrick J Elementary School is pkennedy@bostonpublicschools.org. You can also look at their website at http://bostonpublicschools.org/Page/628

        </EXAMPLE 1>
        
        Previous conversation:
        {self.format_conversation_history()}
        
        Here is the information about a few of the closest schools to the user's home:
        {self.school_context}

        Recall the user's question: "{user_input}" Provide a helpful, conversational response that directly answers their question,
        using the specific schools information provided above. Answer in a short, concise paragraph or less, just like in the example.
        Also make sure you remind the user you can ask follow up questions if they need more information.
        
        <|assistant|>
        """
            
        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides information about Boston schools."},
                {"role": "user", "content": answer_prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
        )
            
        response_text = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": response_text})
        return response_text
    
    def get_response(self, user_input):
        """
        Get a response from the chatbot.
        
        Args:
            user_input (str): The user's input
            
        Returns:
            str: The chatbot's response
        """
        # Add user input to conversation history first
        self.conversation_history.append({"role": "user", "content": user_input})
        
        if self.state == "information_collection":
            # We're still collecting information to find schools
            response, transition = self.handle_information_collection(user_input)
            if transition:
                self.state = "query_mode"
            return response
        else:  # self.state == "query_mode"
            # We already have the schools list and are answering queries about it
            return self.handle_query_mode(user_input)
    
    def reset_conversation(self):
        """Reset the conversation history and state"""
        self.conversation_history = []
        self.state = "information_collection"
        self.user_input_data = None
        self.schools_list = None  # Will store the list of SchoolDetails once retrieved
        self._sent_school_summaries = False

        self.school_summaries = None # Will store formatted school summaries
        self._sent_school_context = False
        self.school_context = None  # Will store full formatted school information