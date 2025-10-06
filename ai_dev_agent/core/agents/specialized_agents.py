"""Specialized agent modes with granular permissions."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
import fnmatch


class Permission(Enum):
    """Permission levels for agent actions."""
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class AgentMode(Enum):
    """Specialized agent operation modes."""
    GENERAL = "general"      # General purpose exploration and implementation
    RESEARCH = "research"    # Read-only research and analysis
    BUILD = "build"         # Full implementation with all permissions
    PLAN = "plan"           # Planning and design without implementation
    DEBUG = "debug"         # Debugging with focused permissions
    REVIEW = "review"       # Code review and analysis


@dataclass
class ToolPermission:
    """Permission configuration for a tool."""

    name: str
    permission: Permission = Permission.ALLOW
    patterns: List[str] = field(default_factory=list)
    max_calls: Optional[int] = None
    require_confirmation: bool = False


@dataclass
class FilePermission:
    """Permission configuration for file operations."""

    read_permission: Permission = Permission.ALLOW
    write_permission: Permission = Permission.ASK
    create_permission: Permission = Permission.ASK
    delete_permission: Permission = Permission.DENY
    patterns_allow: List[str] = field(default_factory=list)
    patterns_deny: List[str] = field(default_factory=list)


@dataclass
class CommandPermission:
    """Permission configuration for shell commands."""

    default_permission: Permission = Permission.ASK
    patterns_allow: List[str] = field(default_factory=list)
    patterns_deny: List[str] = field(default_factory=list)
    dangerous_commands: Set[str] = field(default_factory=lambda: {
        "rm", "del", "format", "fdisk", "dd", "mkfs",
        "shutdown", "reboot", "kill", "pkill", "killall"
    })


@dataclass
class AgentConfig:
    """Configuration for a specialized agent."""

    mode: AgentMode
    name: str
    description: str
    tools: Dict[str, ToolPermission] = field(default_factory=dict)
    file_permissions: FilePermission = field(default_factory=FilePermission)
    command_permissions: CommandPermission = field(default_factory=CommandPermission)
    max_iterations: Optional[int] = None
    temperature: float = 0.7
    model: Optional[str] = None
    system_prompt_addon: Optional[str] = None


class SpecializedAgentManager:
    """Manage specialized agent configurations and permissions."""

    def __init__(self):
        self.agents = self._initialize_default_agents()
        self.current_agent: Optional[AgentConfig] = None
        self.permission_overrides: Dict[str, Permission] = {}

    def _initialize_default_agents(self) -> Dict[AgentMode, AgentConfig]:
        """Initialize default agent configurations."""

        agents = {}

        # General purpose agent
        agents[AgentMode.GENERAL] = AgentConfig(
            mode=AgentMode.GENERAL,
            name="General Assistant",
            description="General purpose exploration and implementation",
            tools={
                "*": ToolPermission("*", Permission.ALLOW),
            },
            file_permissions=FilePermission(
                read_permission=Permission.ALLOW,
                write_permission=Permission.ALLOW,
                create_permission=Permission.ASK
            ),
            command_permissions=CommandPermission(
                default_permission=Permission.ASK
            ),
            system_prompt_addon="You are a general-purpose assistant. Help with any development task."
        )

        # Research agent (read-only)
        agents[AgentMode.RESEARCH] = AgentConfig(
            mode=AgentMode.RESEARCH,
            name="Research Assistant",
            description="Read-only research and code analysis",
            tools={
                "read": ToolPermission("read", Permission.ALLOW),
                "search": ToolPermission("search", Permission.ALLOW),
                "grep": ToolPermission("grep", Permission.ALLOW),
                "list": ToolPermission("list", Permission.ALLOW),
                "write": ToolPermission("write", Permission.DENY),
                "edit": ToolPermission("edit", Permission.DENY),
                "execute": ToolPermission("execute", Permission.DENY),
            },
            file_permissions=FilePermission(
                read_permission=Permission.ALLOW,
                write_permission=Permission.DENY,
                create_permission=Permission.DENY,
                delete_permission=Permission.DENY
            ),
            command_permissions=CommandPermission(
                default_permission=Permission.DENY,
                patterns_allow=["ls", "find", "grep", "cat", "head", "tail", "wc"]
            ),
            temperature=0.3,
            system_prompt_addon="""You are a research assistant with READ-ONLY access.
You can explore and analyze code but CANNOT make any modifications.
Focus on understanding and explaining the codebase."""
        )

        # Build agent (full permissions)
        agents[AgentMode.BUILD] = AgentConfig(
            mode=AgentMode.BUILD,
            name="Build Assistant",
            description="Full implementation with all permissions",
            tools={
                "*": ToolPermission("*", Permission.ALLOW),
            },
            file_permissions=FilePermission(
                read_permission=Permission.ALLOW,
                write_permission=Permission.ALLOW,
                create_permission=Permission.ALLOW,
                delete_permission=Permission.ASK
            ),
            command_permissions=CommandPermission(
                default_permission=Permission.ALLOW,
                patterns_deny=["rm -rf", "format", "fdisk"]
            ),
            temperature=0.7,
            system_prompt_addon="""You are a build assistant with full implementation permissions.
You can create, modify, and execute code to complete development tasks.
Follow best practices and ensure code quality."""
        )

        # Planning agent
        agents[AgentMode.PLAN] = AgentConfig(
            mode=AgentMode.PLAN,
            name="Planning Assistant",
            description="Design and planning without implementation",
            tools={
                "read": ToolPermission("read", Permission.ALLOW),
                "search": ToolPermission("search", Permission.ALLOW),
                "write": ToolPermission("write", Permission.DENY),
                "edit": ToolPermission("edit", Permission.DENY),
            },
            file_permissions=FilePermission(
                read_permission=Permission.ALLOW,
                write_permission=Permission.DENY,
                create_permission=Permission.DENY
            ),
            command_permissions=CommandPermission(
                default_permission=Permission.DENY
            ),
            max_iterations=10,
            temperature=0.8,
            system_prompt_addon="""You are a planning assistant focused on design and architecture.
Analyze requirements, explore the codebase, and create detailed implementation plans.
You cannot modify code directly - provide clear instructions for implementation."""
        )

        # Debug agent
        agents[AgentMode.DEBUG] = AgentConfig(
            mode=AgentMode.DEBUG,
            name="Debug Assistant",
            description="Focused debugging with targeted permissions",
            tools={
                "read": ToolPermission("read", Permission.ALLOW),
                "search": ToolPermission("search", Permission.ALLOW),
                "execute": ToolPermission("execute", Permission.ALLOW),
                "edit": ToolPermission("edit", Permission.ASK),
            },
            file_permissions=FilePermission(
                read_permission=Permission.ALLOW,
                write_permission=Permission.ASK,
                patterns_allow=["*.log", "*.debug", "*.trace"]
            ),
            command_permissions=CommandPermission(
                default_permission=Permission.ASK,
                patterns_allow=["python", "node", "npm", "yarn", "pytest", "jest"]
            ),
            temperature=0.3,
            system_prompt_addon="""You are a debugging assistant focused on finding and fixing issues.
Analyze error messages, trace execution paths, and identify root causes.
Be methodical and thorough in your debugging approach."""
        )

        # Review agent
        agents[AgentMode.REVIEW] = AgentConfig(
            mode=AgentMode.REVIEW,
            name="Review Assistant",
            description="Code review and quality analysis",
            tools={
                "read": ToolPermission("read", Permission.ALLOW),
                "search": ToolPermission("search", Permission.ALLOW),
                "lint": ToolPermission("lint", Permission.ALLOW),
                "test": ToolPermission("test", Permission.ALLOW),
                "write": ToolPermission("write", Permission.DENY),
            },
            file_permissions=FilePermission(
                read_permission=Permission.ALLOW,
                write_permission=Permission.DENY
            ),
            command_permissions=CommandPermission(
                default_permission=Permission.DENY,
                patterns_allow=["pylint", "eslint", "pytest", "jest", "mypy", "tsc"]
            ),
            temperature=0.3,
            system_prompt_addon="""You are a code review assistant focused on quality and best practices.
Analyze code for bugs, security issues, performance problems, and style violations.
Provide constructive feedback and suggest improvements."""
        )

        return agents

    def set_agent_mode(self, mode: AgentMode) -> AgentConfig:
        """Set the current agent mode."""
        if mode not in self.agents:
            raise ValueError(f"Unknown agent mode: {mode}")

        self.current_agent = self.agents[mode]
        return self.current_agent

    def check_tool_permission(
        self,
        tool_name: str,
        arguments: Optional[Dict] = None
    ) -> Tuple[Permission, Optional[str]]:
        """Check if a tool operation is permitted."""

        if not self.current_agent:
            return Permission.ALLOW, None

        # Check for override
        if tool_name in self.permission_overrides:
            return self.permission_overrides[tool_name], None

        # Check specific tool permissions
        for pattern, tool_perm in self.current_agent.tools.items():
            if pattern == "*" or fnmatch.fnmatch(tool_name, pattern):
                if tool_perm.permission == Permission.DENY:
                    return Permission.DENY, f"Tool '{tool_name}' is not allowed in {self.current_agent.mode.value} mode"

                if tool_perm.max_calls and hasattr(self, '_tool_call_counts'):
                    if self._tool_call_counts.get(tool_name, 0) >= tool_perm.max_calls:
                        return Permission.DENY, f"Maximum calls ({tool_perm.max_calls}) reached for '{tool_name}'"

                return tool_perm.permission, None

        # Default to ASK for unknown tools
        return Permission.ASK, None

    def check_file_permission(
        self,
        file_path: str,
        operation: str
    ) -> Tuple[Permission, Optional[str]]:
        """Check if a file operation is permitted."""

        if not self.current_agent:
            return Permission.ALLOW, None

        perms = self.current_agent.file_permissions

        # Check deny patterns first
        for pattern in perms.patterns_deny:
            if fnmatch.fnmatch(file_path, pattern):
                return Permission.DENY, f"File pattern '{pattern}' is denied"

        # Check allow patterns
        for pattern in perms.patterns_allow:
            if fnmatch.fnmatch(file_path, pattern):
                return Permission.ALLOW, None

        # Check operation-specific permissions
        if operation == "read":
            return perms.read_permission, None
        elif operation == "write":
            return perms.write_permission, None
        elif operation == "create":
            return perms.create_permission, None
        elif operation == "delete":
            return perms.delete_permission, None

        return Permission.ASK, None

    def check_command_permission(
        self,
        command: str
    ) -> Tuple[Permission, Optional[str]]:
        """Check if a shell command is permitted."""

        if not self.current_agent:
            return Permission.ALLOW, None

        perms = self.current_agent.command_permissions

        # Check for dangerous commands
        command_parts = command.split()
        if command_parts:
            base_command = command_parts[0]
            if base_command in perms.dangerous_commands:
                return Permission.DENY, f"Command '{base_command}' is considered dangerous"

        # Check deny patterns
        for pattern in perms.patterns_deny:
            if fnmatch.fnmatch(command, pattern):
                return Permission.DENY, f"Command pattern '{pattern}' is denied"

        # Check allow patterns
        for pattern in perms.patterns_allow:
            if fnmatch.fnmatch(command, pattern) or command.startswith(pattern):
                return Permission.ALLOW, None

        return perms.default_permission, None

    def get_allowed_tools(self) -> List[str]:
        """Get list of allowed tools for current agent."""

        if not self.current_agent:
            return []

        allowed = []
        for pattern, tool_perm in self.current_agent.tools.items():
            if tool_perm.permission != Permission.DENY:
                if pattern == "*":
                    return ["all tools"]
                allowed.append(pattern)

        return allowed

    def get_agent_prompt_addon(self) -> Optional[str]:
        """Get agent-specific prompt addon."""

        if not self.current_agent:
            return None

        return self.current_agent.system_prompt_addon

    def override_permission(self, tool_name: str, permission: Permission) -> None:
        """Override permission for a specific tool."""
        self.permission_overrides[tool_name] = permission

    def clear_overrides(self) -> None:
        """Clear all permission overrides."""
        self.permission_overrides.clear()

    def get_agent_summary(self, mode: Optional[AgentMode] = None) -> Dict:
        """Get summary of agent configuration."""

        agent = self.agents.get(mode) if mode else self.current_agent
        if not agent:
            return {}

        return {
            "mode": agent.mode.value,
            "name": agent.name,
            "description": agent.description,
            "allowed_tools": self.get_allowed_tools() if agent == self.current_agent else None,
            "file_permissions": {
                "read": agent.file_permissions.read_permission.value,
                "write": agent.file_permissions.write_permission.value,
                "create": agent.file_permissions.create_permission.value,
                "delete": agent.file_permissions.delete_permission.value,
            },
            "command_permission": agent.command_permissions.default_permission.value,
            "max_iterations": agent.max_iterations,
            "temperature": agent.temperature,
        }