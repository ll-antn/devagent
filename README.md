# DevAgent

> **Proof of Concept**: LLM-powered CLI for development workflows. Interact with your codebase using natural language.

## Installation

```bash
git clone https://github.com/egavrin/devagent.git
cd devagent
pip install -e .
```

## Setup

1. Copy the config template:
```bash
cp .devagent.toml.example .devagent.toml
```

2. Edit `.devagent.toml` with your API details:
```toml
provider = "deepseek"
model = "deepseek-coder"
api_key = "your-api-key"
base_url = "https://api.deepseek.com"
auto_approve_code = false
```

### Setup troubleshooting

#### Error during `pip install...`: Consider using a build backend that supports PEP 660

In case you see this error, try upgrading your `pip` and `setuptools` to the newest version:

```bash
sudo python -m pip install --upgrade pip
sudo python -m pip install --upgrade setuptools
```

#### Unable to find `devagent` command after successfull install

For local installation, `devagent` launcher is created in `$HOME/.local/bin` directory, make sure it is in your `$PATH`.

## Usage

### One-shot queries

```bash
devagent "summarize this repository"
devagent "find all TODO comments"
devagent "explain how the config system works"
```

### Interactive shell

```bash
devagent shell
```

### Development workflow integration

```bash
# Code review
devagent "review my last commit for issues"

# Planning
devagent "plan how to add user authentication"

# Code maintenance
devagent "refactor the database module"
```

### Custom LLM Workflows

DevAgent supports customizing the LLM behavior with three powerful options:

#### `--system`: Custom System Prompt

Extend or customize the system prompt to define LLM behavior and role:

```bash
# Inline system prompt
devagent query --system "You are a security expert" "review this code"

# System prompt from file
devagent query --system prompts/code_reviewer.md "analyze main.py"
```

#### `--prompt`: User Prompt from File or String

Load user prompts from files or pass them inline:

```bash
# Prompt from file
devagent --prompt input.txt

# Combine with system context
devagent --system "Explain like I'm 5" --prompt question.txt
```

#### `--format`: Structured Output with JSON Schema

Specify the output format using JSON Schema:

```bash
# Request structured JSON output
devagent query \
  --system prompts/reviewer.md \
  --prompt code.py \
  --format schemas/code_review.json
```

**Example format schema** (`schemas/code_review.json`):
```json
{
  "type": "object",
  "properties": {
    "issues": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "severity": {"type": "string"},
          "line": {"type": "integer"},
          "description": {"type": "string"}
        }
      }
    },
    "summary": {"type": "string"}
  },
  "required": ["issues", "summary"]
}
```

**Key Features:**
- **File auto-detection**: If argument is a file path, reads content; otherwise uses as literal string
- **Generic mechanism**: Works for any workflow - code review, data extraction, test generation, etc.
- **Composable**: Combine with existing `--plan`, `--direct` flags

## Testing

```bash
pip install -e .[dev]
pytest
```

That's it. This PoC demonstrates how LLMs can assist development workflows while keeping humans in control.
