"""
Utility functions for conflict dataset generation.
"""

import json
import re
from typing import List, Dict


def load_jsonl_response(text: str) -> List[Dict]:
    """
    Extract and parse JSONL from LLM output text.
    Handles various formats including:
    - Plain JSONL
    - JSONL in markdown code blocks with ```jsonl
    
    Args:
        text: String containing JSONL data
    
    Returns:
        List of parsed JSON objects
    """
    # First try parsing the entire text as JSONL
    try:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError:
        pass

    # Look for JSONL in code blocks
    code_block_patterns = [
        r"```jsonl\n(.*?)\n```",  # ```jsonl ... ```
        r"```jsonl\\n(.*?)\\n```",  # escaped newlines
        r"```json\n(.*?)\n```",  # ```json ... ```
        r"```\n(.*?)\n```",  # ``` ... ```
    ]

    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            for match in matches:
                try:
                    res = [
                        json.loads(line) 
                        for line in match.splitlines() 
                        if line.strip()
                    ]
                    if res:
                        return res
                except json.JSONDecodeError:
                    continue

    # Try to find any JSON objects line by line
    results = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    if results:
        return results

    raise json.JSONDecodeError("No valid JSONL found in the text", text, 0)


def load_json_response(text: str) -> Dict:
    """
    Extract and parse JSON from LLM output text.
    
    Args:
        text: String containing JSON data
    
    Returns:
        Parsed JSON object
    """
    # First try parsing the entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Look for JSON in code blocks
    code_block_patterns = [
        r"```json\n(.*?)\n```",
        r"```\n(.*?)\n```",
    ]

    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

    # Look for content between curly braces
    brace_content = re.findall(r"\{.*\}", text, re.DOTALL)
    if brace_content:
        for content in brace_content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                continue

    raise json.JSONDecodeError("No valid JSON found in the text", text, 0)


class FileGenerationError(Exception):
    """Raised when document generation fails."""
    pass

