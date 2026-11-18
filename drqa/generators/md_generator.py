"""
Markdown document generator for conflict dataset.
Generates professional markdown reports.
"""

import os
import re
from pathlib import Path
from typing import Dict

from llm import prompt_llm, DEFAULT_MODEL


def generate_markdown_document(
    topic: str,
    output_path: str = None,
    word_limit: int = 2000,
    existing_files: list = None,
    company_info: Dict = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Generate a markdown document about a topic.

    Args:
        topic: The topic to include in the document
        output_path: The directory where the markdown should be saved
        word_limit: Maximum number of words in the document content
        existing_files: List of existing files to avoid redundancy
        company_info: Dictionary containing company-related context
        model: The model to use for generation

    Returns:
        The path where the markdown was saved (or content if no output_path)
    """
    print("Generating markdown document...")

    markdown_prompt = f"""
    You are a markdown expert who creates professional documents with proper structure, tables, and formatting. Create a complete markdown document given the following requirements.
    
    {topic}
    
    The document should:
    1. Use proper markdown headings (# for main title, ## for sections, etc.)
    2. Be approximately {word_limit} words in total
    3. Include at least one table with relevant data
    4. Use proper markdown formatting for:
       - Lists (both ordered and unordered)
       - Code blocks (where appropriate)
       - Blockquotes (for important points)
       - Links (where relevant)
    5. Use professional terminology and include relevant examples
    6. Only use realistic names for people
    7. Use a random name for the document author
    8. Include a title and author at the beginning
    9. Never ever include file paths in the document
    """

    if existing_files and len(existing_files) > 0:
        markdown_prompt += "\nThe following files already exist in the directory:\n"
        for file_path in existing_files:
            markdown_prompt += f"- {Path(file_path).name}\n"
        markdown_prompt += "\nPlease create content that avoids redundancy with these existing files. Write about something else."

    if company_info and isinstance(company_info, dict):
        markdown_prompt += "\nCompany Context:\n"
        for key, value in company_info.items():
            markdown_prompt += f"- {key}: {value}\n"
        markdown_prompt += (
            "\nPlease keep this company context in mind when creating the document."
        )

    markdown_prompt += "\nReturn only the complete markdown content."

    markdown_content = (
        prompt_llm(markdown_prompt, model=model).lstrip("```markdown\n").rstrip("```")
    )

    # Use the title as file name
    try:
        filename = re.match(r"# (.*)", markdown_content).group(1)
    except Exception:
        # Use the first 100 characters of the content as the filename
        filename = markdown_content[:100]

    if not filename.lower().endswith(".md"):
        filename += ".md"
    filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
    filename = filename.replace(" ", "_")

    # Determine the final output path
    if output_path is None:
        # Return content directly
        return markdown_content
    else:
        # Always use the directory/filename combination
        output_path_obj = Path(output_path)
        # Ensure the output directory exists
        os.makedirs(output_path_obj, exist_ok=True)
        final_output_path = str(output_path_obj / filename)

    # Write the markdown content to file
    with open(final_output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"Markdown document successfully generated at: {final_output_path}")
    return final_output_path

