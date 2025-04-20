import anthropic
import os
import json
import time
from dotenv import load_dotenv
from pathlib import Path
import random
import uuid

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

OUTPUT_DIR = Path("synthetic_evals")
OUTPUT_DIR.mkdir(exist_ok=True)

system_prompt = """
You are an expert assistant that generates synthetic, structured conversation flows for evaluating school enrollment chatbots. Use the following JSON schema and generate realistic conversations.

Each conversation should simulate a multi-turn dialogue between a parent and the assistant, with expected assistant behavior and metadata.

Return a single conversation flow in JSON format.
"""

timeline_options = ["enrolling this fall", "next school year", "mid-year transfer", "exploring options early"]
pref_options = [["bilingual programs"], ["bilingual programs", "after-school care"], ["bilingual programs", "STEM"]]
special_options = [["transportation needs"], ["transportation needs", "IEP support"], ["transportation needs", "ESL support"]]

with open("eval_schema.json", "r") as schema_file:
    schema_structure = schema_file.read()

def generate_user_prompt():
    timeline = random.choice(timeline_options)
    preferences = random.choice(pref_options)
    special = random.choice(special_options)

    return f"""
Generate a synthetic conversation_flow_schema JSON object.

Scenario:
- Neighborhood: Roxbury
- Grade level: elementary
- Preferences: {', '.join(preferences)}
- Special considerations: {', '.join(special)}
- Enrollment timeline: {timeline}

Include clarifying questions and appropriate assistant behavior. Follow this JSON structure exactly:

{schema_structure}
""" 

for i in range(1, 51):
    print(f"Generating conversation {i}/50...")

    user_prompt = generate_user_prompt()

    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000, # can change
            temperature=0.7,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        response_text = response.content[0].text.strip()

        try:
            # Parse and save valid JSON
            data = json.loads(response_text)
            file_id = f"{i:03}"
            with open(OUTPUT_DIR / f"eval_{file_id}.json", "w") as f:
                json.dump(data, f, indent=2)
            print(f"✅ Saved eval_{file_id}.json")

        except json.JSONDecodeError:
            # Save as fallback .txt
            file_id = f"{i:03}"
            with open(OUTPUT_DIR / f"eval_{file_id}_raw.txt", "w") as f:
                f.write(response_text)
            print(f"⚠️ Saved fallback for eval_{file_id} (invalid JSON)")

    except Exception as e:
        print(f"❌ Error generating conversation {i}: {e}")

    time.sleep(1.5)  # Rate limit safety
