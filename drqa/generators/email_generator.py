"""
Email document generator for conflict dataset.
Generates professional email threads in JSONL format.
"""

from datetime import datetime
import os
import json
import random
import string
from typing import Dict

from llm import prompt_llm, DEFAULT_MODEL
from utils import load_jsonl_response, FileGenerationError


def generate_random_password(length: int = 12) -> str:
    """Generate a random password with unicode characters."""
    chars = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?"
    return "".join(random.choice(chars) for _ in range(length))


def generate_email_document(
    topic: str,
    persona: Dict,
    output_dir: str = None,
    word_limit: int = 500,
    company_info: Dict = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Generate an email document about a topic.

    Args:
        topic: The topic to include in the emails
        persona: The persona to use for the emails
        output_dir: The directory where the emails should be saved
        word_limit: Approximate word limit for all emails combined
        company_info: Dictionary containing company-related context
        model: The model to use for generation

    Returns:
        The path where the emails were saved
    """
    credentials = {
        "type": "user",
        "username": f"{persona['first_name']}.{persona['last_name']}",
        "first_name": persona["first_name"],
        "last_name": persona["last_name"],
        "email": persona["email"],
        "password": generate_random_password(),
    }

    print("Generating email document...")

    email_prompt = f"""
    You are an email expert who creates professional emails with proper structure and formatting.
    Create a JSONL file that contains a series of email messages in the mail boxes of persona: {persona['first_name']} {persona['last_name']}, role: {persona['role']}, department: {persona['department']}, responsibilities: {persona['responsibilities']}.
    The emails need to cover the following information:
    
    {topic}
    
    The emails should:
    1. Be realistic email conversations that mention the answer to the question inherently
    2. Follow this exact format for each email:
       {{
           "type": "email",
           "id": "email_001",
           "from": "sender@company.com",
           "from_name": "Sender Name",
           "to": ["recipient@company.com"],
           "cc": [],
           "subject": "Email Subject",
           "date": "2025-01-20T09:00:00-05:00",
           "body": "Email body content...",
           "folder": "inbox",
           "read": false,
           "attachments": []
       }}
    3. Have the persona as the sender of one of the recipients (either to or cc)
    4. Be approximately {word_limit} words in total across all emails
    5. Use realistic names, email addresses, and subjects based on the topic
    6. Email IDs should be sequential (email_001, email_002, etc.)
    7. Dates should be realistic and recent (within the last few months)
    8. Folders can be: inbox, sent, drafts, spam, trash
    9. Some emails should be read (true), some unread (false)

    Consider generating:
    - Emails between team members discussing around the topic
    - Emails from management to employees asking for specific information on the topic
    - Internal announcements or updates

    Wrap the email content in a JSONL code block, e.g. ```jsonl

[EMAIL CONTENT]

```.
    """

    if company_info and isinstance(company_info, dict):
        email_prompt += "\nCompany Context:\n"
        for key, value in company_info.items():
            email_prompt += f"- {key}: {value}\n"
        email_prompt += (
            "\nPlease keep this company context in mind when creating the emails."
        )

    email_prompt += "\nReturn only the complete email content in a JSONL format."
    response = prompt_llm(email_prompt, model=model)
    email_content = load_jsonl_response(response)

    if len(email_content) == 0:
        raise FileGenerationError("Failed to generate an email document.")

    # Save the email content
    if output_dir:
        output_path = os.path.join(
            output_dir,
            f"email_document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        )
        with open(output_path, "w") as f:
            # Write persona credentials first
            f.write(json.dumps(credentials) + "\n")

            # Write each email as a separate line
            for item in email_content:
                # Skip non-dict items (e.g., malformed JSON parsing)
                if not isinstance(item, dict):
                    continue
                    
                # Ensure email has required fields
                if item.get("type") == "email":
                    if "id" not in item:
                        item["id"] = f"email_{len(email_content):03d}"
                    if "read" not in item:
                        item["read"] = random.choice([True, False])
                    if "folder" not in item:
                        item["folder"] = random.choice(["inbox", "sent", "drafts"])
                    # default attachments to empty list
                    item["attachments"] = []

                f.write(json.dumps(item) + "\n")

        return output_path
    
    # If no output_dir, return the content as string
    lines = [json.dumps(credentials)]
    for item in email_content:
        if isinstance(item, dict):
            lines.append(json.dumps(item))
    return "\n".join(lines)

