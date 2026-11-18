"""
Chat document generator for conflict dataset.
Generates Mattermost-style chat logs in JSONL format.
"""

from datetime import datetime
import os
import json
from typing import Dict

from llm import prompt_llm, DEFAULT_MODEL
from utils import load_jsonl_response, FileGenerationError


def generate_mattermost_chat(
    topic: str,
    output_dir: str = None,
    word_limit: int = 500,
    company_info: Dict = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Generate a MatterMost chat document about a topic.

    Args:
        topic: The topic to include in the chat
        output_dir: The directory where the chat should be saved
        word_limit: Approximate word limit for the chat
        company_info: Dictionary containing company-related context
        model: The model to use for generation

    Returns:
        The path where the chat was saved
    """
    print("Generating MatterMost chat...")

    chat_prompt = f"""
    You are a MatterMost chat expert who creates professional chats with proper structure, tables, and formatting. Create a jsonl file that contains a series of chat messages in channels. The messages need to cover the following information:
    
    {topic}
    
    The chat should:
    1. Be discussions between reasonable users on the topic and mentions the answer to the question inherently
    2. Contains realistic messages to team/channel (teams for a group of people that work together, channels for projects), e.g. {{"type": "post", "post": {{"team": "marketing", "channel": "campaigns", "user": "samantha.lee", "message": "Here's the draft of our Q3 campaign plan:", "create_at": 1620500000000}}}}
    3. Contains all the necessary team, channel and user settings for all posts, e.g. {{"type": "channel", "channel": {{"team": "marketing", "name": "campaigns", "display_name": "Campaigns", "type": "O", "header": "Marketing campaign planning", "purpose": "Channel for planning marketing campaigns"}}}}
    4. Be approximately {word_limit} words in total
    5. Use realistic names for people/teams/channels and create realistic conversations based on the topic
    6. Contains a version object at the beginning of the file, e.g. {{"version": 1.0}}
    7. Teams, Channels, and Users must appear in that order, and must appear before posts.
    8. Users include credentials information and the list of teams they belong to

    Consider generating:
    - Chats between customers/partners and support/sales/product teams
    - Chats between team members
    - Chats between employees and management

    Wrap the chat content in a JSONL code block, e.g. ```jsonl

[CHAT CONTENT]

```.
    """

    if company_info and isinstance(company_info, dict):
        chat_prompt += "\nCompany Context:\n"
        for key, value in company_info.items():
            chat_prompt += f"- {key}: {value}\n"
        chat_prompt += (
            "\nPlease keep this company context in mind when creating the document."
        )

    chat_prompt += "\nReturn only the complete chat content in a JSONL format."
    response = prompt_llm(chat_prompt, model=model)
    chat_content = load_jsonl_response(response)

    if len(chat_content) == 0:
        raise FileGenerationError("Failed to generate a chat document.")

    # Save the chat content
    if output_dir:
        output_path = os.path.join(
            output_dir,
            f"mattermost_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        )
        with open(output_path, "w") as f:
            # Write each JSON object as a separate line
            for item in chat_content:
                # Skip non-dict items (e.g., malformed JSON parsing)
                if not isinstance(item, dict):
                    continue
                # Cast version value to a number
                if isinstance(item.get("version"), str):
                    item["version"] = float(item["version"])
                f.write(json.dumps(item) + "\n")

        return output_path
    
    # If no output_dir, return the content as string
    lines = []
    for item in chat_content:
        if isinstance(item, dict):
            if isinstance(item.get("version"), str):
                item["version"] = float(item["version"])
            lines.append(json.dumps(item))
    return "\n".join(lines)

