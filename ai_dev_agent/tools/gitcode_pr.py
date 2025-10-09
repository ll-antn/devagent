"""Tool to fetch and parse Pull Request data from GitCode API."""
import os
import urllib.request
import json
from typing import Any, Mapping
from pathlib import Path

from .registry import ToolSpec, ToolContext, registry

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas" / "tools"


def _convert_to_standard_diff(api_response: dict) -> list[dict[str, str]]:
    """
    Convert GitCode API response to standard diff format.
    
    Args:
        api_response: Raw response from GitCode API
        
    Returns:
        List of dicts with 'file' and 'diff' keys
    """
    result = []
    
    if "diffs" not in api_response:
        return result
    
    for diff_item in api_response["diffs"]:
        # Extract file path
        file_path = diff_item.get("statistic", {}).get("path", "unknown")
        
        # Build standard diff format from the content
        diff_lines = []
        
        # Add diff header
        old_path = diff_item.get("statistic", {}).get("old_path", file_path)
        new_path = diff_item.get("statistic", {}).get("new_path", file_path)
        diff_lines.append(f"--- a/{old_path}")
        diff_lines.append(f"+++ b/{new_path}")
        
        # Process text content
        content = diff_item.get("content", {})
        text_lines = content.get("text", [])
        
        for line_item in text_lines:
            line_content = line_item.get("line_content", "")
            line_type = line_item.get("type", "")
            
            # Handle different line types
            if line_type == "match":
                # Hunk header (e.g., @@ -0,0 +1,29 @@)
                diff_lines.append(line_content)
            elif line_type == "new":
                # Added line
                diff_lines.append(f"+{line_content}")
            elif line_type == "old":
                # Removed line
                diff_lines.append(f"-{line_content}")
            elif line_type == "context" or line_type == "":
                # Context line (unchanged)
                diff_lines.append(f" {line_content}")
        
        # Join lines into a single diff string
        diff_str = "\n".join(diff_lines)
        
        result.append({
            "file": file_path,
            "diff": diff_str,
            "added_lines": diff_item.get("added_lines", 0),
            "removed_lines": diff_item.get("remove_lines", 0)
        })
    
    return result


def get_gitcode_pr(payload: Mapping[str, Any], context: ToolContext) -> Mapping[str, Any]:
    """
    Fetch Pull Request data from GitCode API.
    
    Args:
        payload: Contains 'owner', 'repo', 'number', and optional 'token' fields
        context: Tool execution context
        
    Returns:
        Dict with 'files' list containing file paths and diffs
    """
    owner = payload.get("owner", "")
    repo = payload.get("repo", "")
    number = payload.get("number", 0)
    # Token priority: payload > environment variable
    token = payload.get("token") or os.environ.get("GITCODE_TOKEN")
    
    if not owner or not repo or not number:
        return {
            "error": "Missing required fields: owner, repo, and number",
            "files": []
        }
    
    # Construct API URL
    url = f"https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}/files.json"
    
    try:
        # Make HTTP request
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        
        # Add authentication token if provided
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        # Check for API errors
        if "code" in data and data["code"] != 0:
            return {
                "error": f"API returned error code: {data['code']}",
                "files": []
            }
        
        # Convert to standard diff format
        files = _convert_to_standard_diff(data)
        
        # Add summary information
        summary = {
            "total_files": data.get("count", 0),
            "added_lines": data.get("added_lines", 0),
            "removed_lines": data.get("remove_lines", 0),
            "base_sha": data.get("diff_refs", {}).get("base_sha", ""),
            "head_sha": data.get("diff_refs", {}).get("head_sha", "")
        }
        
        return {
            "files": files,
            "summary": summary
        }
        
    except urllib.error.HTTPError as e:
        return {
            "error": f"HTTP error {e.code}: {e.reason}",
            "files": []
        }
    except urllib.error.URLError as e:
        return {
            "error": f"URL error: {e.reason}",
            "files": []
        }
    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse JSON response: {str(e)}",
            "files": []
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "files": []
        }


# Register the tool
registry.register(
    ToolSpec(
        name="gitcode_pr",
        handler=get_gitcode_pr,
        request_schema_path=SCHEMA_DIR / "gitcode_pr.request.json",
        response_schema_path=SCHEMA_DIR / "gitcode_pr.response.json",
        description=(
            "Fetch Pull Request data from GitCode API with authentication.\n"
            "Returns file changes in standard diff format.\n"
            "Requires authentication via 'token' parameter or GITCODE_TOKEN env var.\n"
            "Example: gitcode_pr(owner='nadolskiyanton', repo='AIReviewTest', number=1)\n"
            "Use this to analyze PR changes, review code, and understand modifications."
        ),
        display_name="GitCode PR Fetcher",
        category="vcs",
    )
)
