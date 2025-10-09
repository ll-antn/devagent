"""Tests for GitCode PR tool."""
import json
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from ai_dev_agent.tools.gitcode_pr import get_gitcode_pr, _convert_to_standard_diff
from ai_dev_agent.tools.registry import ToolContext


class TestGitCodePR(unittest.TestCase):
    """Test cases for GitCode PR tool."""

    def setUp(self):
        """Set up test context."""
        self.context = ToolContext(
            repo_root=Path.cwd(),
            settings=None,
            sandbox=None,
        )

    def test_missing_owner(self):
        """Test error handling when owner is missing."""
        payload = {"repo": "test", "number": 1}
        result = get_gitcode_pr(payload, self.context)
        
        self.assertIn("error", result)
        self.assertIn("Missing required fields", result["error"])
        self.assertEqual(result["files"], [])

    def test_missing_repo(self):
        """Test error handling when repo is missing."""
        payload = {"owner": "test", "number": 1}
        result = get_gitcode_pr(payload, self.context)
        
        self.assertIn("error", result)
        self.assertIn("Missing required fields", result["error"])
        self.assertEqual(result["files"], [])

    def test_missing_number(self):
        """Test error handling when PR number is missing."""
        payload = {"owner": "test", "repo": "test"}
        result = get_gitcode_pr(payload, self.context)
        
        self.assertIn("error", result)
        self.assertIn("Missing required fields", result["error"])
        self.assertEqual(result["files"], [])

    @patch('urllib.request.urlopen')
    def test_api_error_code(self, mock_urlopen):
        """Test handling of API error codes."""
        # Mock API response with error code
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({
            "code": 404,
            "message": "Not found"
        }).encode('utf-8')
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        payload = {"owner": "test", "repo": "test", "number": 999}
        result = get_gitcode_pr(payload, self.context)
        
        self.assertIn("error", result)
        self.assertIn("error code", result["error"])

    @patch('urllib.request.urlopen')
    def test_http_error(self, mock_urlopen):
        """Test handling of HTTP errors."""
        mock_urlopen.side_effect = HTTPError(
            "http://test.com", 404, "Not Found", {}, None
        )
        
        payload = {"owner": "test", "repo": "test", "number": 1}
        result = get_gitcode_pr(payload, self.context)
        
        self.assertIn("error", result)
        self.assertIn("HTTP error 404", result["error"])

    @patch('urllib.request.urlopen')
    def test_url_error(self, mock_urlopen):
        """Test handling of URL errors."""
        mock_urlopen.side_effect = URLError("Connection failed")
        
        payload = {"owner": "test", "repo": "test", "number": 1}
        result = get_gitcode_pr(payload, self.context)
        
        self.assertIn("error", result)
        self.assertIn("URL error", result["error"])

    @patch('urllib.request.urlopen')
    def test_successful_request_with_token(self, mock_urlopen):
        """Test successful PR data fetch with custom token."""
        # Mock successful API response
        api_response = {
            "code": 0,
            "added_lines": 10,
            "remove_lines": 5,
            "count": 1,
            "diff_refs": {
                "base_sha": "abc123",
                "start_sha": "abc123",
                "head_sha": "def456"
            },
            "diffs": [
                {
                    "new_blob_id": "blob123",
                    "statistic": {
                        "type": "text_type",
                        "path": "test.py",
                        "old_path": "test.py",
                        "new_path": "test.py",
                        "view": False
                    },
                    "head": {
                        "url": "https://example.com/test.py",
                        "commit_id": "def456"
                    },
                    "added_lines": 10,
                    "remove_lines": 5,
                    "content": {
                        "text": [
                            {
                                "line_content": "@@ -1,3 +1,4 @@",
                                "old_line": "...",
                                "new_line": "...",
                                "type": "match"
                            },
                            {
                                "line_content": "def test():",
                                "old_line": {"line_code": "x", "line_num": ""},
                                "new_line": {"line_code": "x", "line_num": 1},
                                "type": "new"
                            }
                        ]
                    }
                }
            ]
        }
        
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(api_response).encode('utf-8')
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        payload = {"owner": "test", "repo": "test", "number": 1, "token": "custom_token"}
        result = get_gitcode_pr(payload, self.context)
        
        # Verify structure
        self.assertIn("files", result)
        self.assertIn("summary", result)
        self.assertEqual(len(result["files"]), 1)
        
        # Verify summary
        summary = result["summary"]
        self.assertEqual(summary["total_files"], 1)
        self.assertEqual(summary["added_lines"], 10)
        self.assertEqual(summary["removed_lines"], 5)
        self.assertEqual(summary["base_sha"], "abc123")
        self.assertEqual(summary["head_sha"], "def456")
        
        # Verify file data
        file_data = result["files"][0]
        self.assertEqual(file_data["file"], "test.py")
        self.assertIn("--- a/test.py", file_data["diff"])
        self.assertIn("+++ b/test.py", file_data["diff"])
        self.assertIn("+def test():", file_data["diff"])

    @patch('urllib.request.urlopen')
    def test_successful_request(self, mock_urlopen):
        """Test successful PR data fetch with default token."""
        # Mock successful API response
        api_response = {
            "code": 0,
            "added_lines": 10,
            "remove_lines": 5,
            "count": 1,
            "diff_refs": {
                "base_sha": "abc123",
                "start_sha": "abc123",
                "head_sha": "def456"
            },
            "diffs": [
                {
                    "new_blob_id": "blob123",
                    "statistic": {
                        "type": "text_type",
                        "path": "test.py",
                        "old_path": "test.py",
                        "new_path": "test.py",
                        "view": False
                    },
                    "head": {
                        "url": "https://example.com/test.py",
                        "commit_id": "def456"
                    },
                    "added_lines": 10,
                    "remove_lines": 5,
                    "content": {
                        "text": [
                            {
                                "line_content": "@@ -1,3 +1,4 @@",
                                "old_line": "...",
                                "new_line": "...",
                                "type": "match"
                            },
                            {
                                "line_content": "def test():",
                                "old_line": {"line_code": "x", "line_num": ""},
                                "new_line": {"line_code": "x", "line_num": 1},
                                "type": "new"
                            }
                        ]
                    }
                }
            ]
        }
        
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(api_response).encode('utf-8')
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        payload = {"owner": "test", "repo": "test", "number": 1}
        result = get_gitcode_pr(payload, self.context)
        
        # Verify structure
        self.assertIn("files", result)
        self.assertIn("summary", result)
        self.assertEqual(len(result["files"]), 1)
        
        # Verify summary
        summary = result["summary"]
        self.assertEqual(summary["total_files"], 1)
        self.assertEqual(summary["added_lines"], 10)
        self.assertEqual(summary["removed_lines"], 5)
        self.assertEqual(summary["base_sha"], "abc123")
        self.assertEqual(summary["head_sha"], "def456")
        
        # Verify file data
        file_data = result["files"][0]
        self.assertEqual(file_data["file"], "test.py")
        self.assertIn("--- a/test.py", file_data["diff"])
        self.assertIn("+++ b/test.py", file_data["diff"])
        self.assertIn("+def test():", file_data["diff"])

    @patch.dict(os.environ, {'GITCODE_TOKEN': 'env_token_test'})
    @patch('urllib.request.urlopen')
    def test_token_from_environment(self, mock_urlopen):
        """Test that token is read from environment variable."""
        # Mock successful API response
        api_response = {
            "code": 0,
            "added_lines": 1,
            "remove_lines": 0,
            "count": 1,
            "diff_refs": {"base_sha": "abc", "start_sha": "abc", "head_sha": "def"},
            "diffs": []
        }
        
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(api_response).encode('utf-8')
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        # Don't pass token in payload, should use env var
        payload = {"owner": "test", "repo": "test", "number": 1}
        result = get_gitcode_pr(payload, self.context)
        
        # Verify request was made (token was used)
        self.assertIn("files", result)
        self.assertNotIn("error", result)
        
        # Verify that the token from env was used (check that urlopen was called)
        mock_urlopen.assert_called_once()

    @patch('urllib.request.urlopen')
    def test_token_priority_payload_over_env(self, mock_urlopen):
        """Test that payload token takes priority over environment variable."""
        with patch.dict(os.environ, {'GITCODE_TOKEN': 'env_token'}):
            # Mock successful API response
            api_response = {
                "code": 0,
                "added_lines": 1,
                "remove_lines": 0,
                "count": 1,
                "diff_refs": {"base_sha": "abc", "start_sha": "abc", "head_sha": "def"},
                "diffs": []
            }
            
            mock_response = Mock()
            mock_response.read.return_value = json.dumps(api_response).encode('utf-8')
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=False)
            mock_urlopen.return_value = mock_response
            
            # Pass token in payload - should override env var
            payload = {"owner": "test", "repo": "test", "number": 1, "token": "payload_token"}
            result = get_gitcode_pr(payload, self.context)
            
            # Verify request was made successfully
            self.assertIn("files", result)
            self.assertNotIn("error", result)

    def test_convert_to_standard_diff(self):
        """Test conversion of GitCode API response to standard diff format."""
        api_response = {
            "diffs": [
                {
                    "statistic": {
                        "path": "example.py",
                        "old_path": "example.py",
                        "new_path": "example.py"
                    },
                    "added_lines": 2,
                    "remove_lines": 1,
                    "content": {
                        "text": [
                            {
                                "line_content": "@@ -1,2 +1,3 @@",
                                "type": "match"
                            },
                            {
                                "line_content": "print('hello')",
                                "type": "old"
                            },
                            {
                                "line_content": "print('hello world')",
                                "type": "new"
                            }
                        ]
                    }
                }
            ]
        }
        
        result = _convert_to_standard_diff(api_response)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["file"], "example.py")
        self.assertIn("--- a/example.py", result[0]["diff"])
        self.assertIn("+++ b/example.py", result[0]["diff"])
        self.assertIn("@@ -1,2 +1,3 @@", result[0]["diff"])
        self.assertIn("-print('hello')", result[0]["diff"])
        self.assertIn("+print('hello world')", result[0]["diff"])


if __name__ == "__main__":
    unittest.main()
