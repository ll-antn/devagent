# GitCode PR Tool

A tool to fetch and parse Pull Request data from the GitCode API.

## Features

- Fetches PR file changes from GitCode API
- Converts GitCode's custom diff format to standard unified diff format
- Can be used standalone as a library or through the ReAct agent
- Provides comprehensive PR summary information

## Installation

The tool is automatically registered when you import the `ai_dev_agent.tools` module.

## Authentication

**Authentication is required.** The tool supports two ways to provide authentication (in order of priority):

1. **Payload parameter**: Pass `token` in the payload
2. **Environment variable**: Set `GITCODE_TOKEN` environment variable

Example:
```bash
export GITCODE_TOKEN="your_token_here"
```

If neither is provided, the API request may fail or have limited access.

## Usage

### As a Standalone Library

```python
from pathlib import Path
from ai_dev_agent.tools.gitcode_pr import get_gitcode_pr
from ai_dev_agent.tools.registry import ToolContext

# Create a minimal tool context
context = ToolContext(
    repo_root=Path.cwd(),
    settings=None,
    sandbox=None,
)

# Fetch PR data
payload = {
    "owner": "nadolskiyanton",
    "repo": "AIReviewTest",
    "number": 1,
    # Optional: provide auth token (uses default if not specified)
    # "token": "your_auth_token_here"
}

result = get_gitcode_pr(payload, context)

# Access the results
if "error" not in result or not result["error"]:
    summary = result["summary"]
    files = result["files"]
    
    for file_change in files:
        print(f"File: {file_change['file']}")
        print(f"Diff:\n{file_change['diff']}")
```

### Through ReAct Agent

Simply ask the agent:
```
Get PR number 1 from nadolskiyanton/AIReviewTest repo
```

The agent will automatically use the `gitcode_pr` tool to fetch the data.

### Using the Tool Registry

```python
from ai_dev_agent.tools import registry, ToolContext
from pathlib import Path

context = ToolContext(
    repo_root=Path.cwd(),
    settings=None,
    sandbox=None,
)

result = registry.invoke(
    "gitcode_pr",
    {"owner": "user", "repo": "repo", "number": 1},
    context
)
```

## API

### Input Parameters

- **owner** (string, required): Repository owner (username or organization)
- **repo** (string, required): Repository name
- **number** (integer, required): Pull Request number (minimum: 1)
- **token** (string, optional): Authentication token for GitCode API
  - If not provided, reads from `GITCODE_TOKEN` environment variable
  - At least one authentication method should be provided

### Output Format

```python
{
    "files": [
        {
            "file": "path/to/file.py",
            "diff": "--- a/path/to/file.py\n+++ b/path/to/file.py\n...",
            "added_lines": 10,
            "removed_lines": 5
        }
    ],
    "summary": {
        "total_files": 1,
        "added_lines": 10,
        "removed_lines": 5,
        "base_sha": "abc123...",
        "head_sha": "def456..."
    },
    "error": "Error message if any"  # Optional
}
```

## Example

Run the included example:

```bash
python examples/gitcode_pr_example.py
```

This will fetch and display PR #1 from the nadolskiyanton/AIReviewTest repository.

## Error Handling

The tool handles various error scenarios:
- Missing required parameters
- API errors (404, 500, etc.)
- Network errors
- Invalid JSON responses

All errors are returned in the `error` field of the response.

## API Endpoint

The tool uses the GitCode API endpoint:
```
GET https://api.gitcode.com/api/v5/repos/:owner/:repo/pulls/:number/files.json
```

## Testing

Run the test suite:

```bash
python -m unittest tests.test_gitcode_pr -v
```
