#!/usr/bin/env python3
"""
Example demonstrating standalone usage of the GitCode PR tool.

This shows how to use the tool as a library without the ReAct agent.

Authentication (required):
- Set GITCODE_TOKEN environment variable, or
- Pass token in the payload
"""
import os
from pathlib import Path
from ai_dev_agent.tools.gitcode_pr import get_gitcode_pr
from ai_dev_agent.tools.registry import ToolContext


def main():
    """Fetch and display PR data from GitCode."""
    # Create a minimal tool context (required by the tool interface)
    context = ToolContext(
        repo_root=Path.cwd(),
        settings=None,
        sandbox=None,
    )
    
    # Example: Fetch PR #1 from nadolskiyanton/AIReviewTest
    payload = {
        "owner": "nadolskiyanton",
        "repo": "AIReviewTest",
        "number": 1,
        # Token is optional:
        # 1. Pass explicitly: "token": "your_auth_token_here"
        # 2. Set GITCODE_TOKEN env var
        # 3. Falls back to default
    }
    
    print("Fetching PR data from GitCode...")
    print(f"Owner: {payload['owner']}")
    print(f"Repo: {payload['repo']}")
    print(f"PR Number: {payload['number']}")
    
    # Show authentication info
    env_token = os.environ.get("GITCODE_TOKEN")
    payload_token = payload.get("token")
    if payload_token:
        print("Using token from payload parameter")
    elif env_token:
        print("Using token from GITCODE_TOKEN env var")
    else:
        print("WARNING: No authentication token provided!")
    
    print("-" * 80)
    
    # Call the tool
    result = get_gitcode_pr(payload, context)
    
    # Check for errors
    if "error" in result and result["error"]:
        print(f"Error: {result['error']}")
        return 1
    
    # Display summary
    if "summary" in result:
        summary = result["summary"]
        print("\n=== PR Summary ===")
        print(f"Total files changed: {summary.get('total_files', 0)}")
        print(f"Total lines added: {summary.get('added_lines', 0)}")
        print(f"Total lines removed: {summary.get('removed_lines', 0)}")
        print(f"Base SHA: {summary.get('base_sha', 'N/A')}")
        print(f"Head SHA: {summary.get('head_sha', 'N/A')}")
    
    # Display file changes
    files = result.get("files", [])
    print(f"\n=== Changed Files ({len(files)}) ===")
    
    for file_change in files:
        file_path = file_change.get("file", "unknown")
        added = file_change.get("added_lines", 0)
        removed = file_change.get("removed_lines", 0)
        
        print(f"\n📄 {file_path}")
        print(f"   +{added} -{removed}")
        print("\n--- Diff ---")
        print(file_change.get("diff", "No diff available"))
        print("-" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())
