"""Command line interface for the development agent."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import List, Optional, Tuple

import click

from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.core.utils.logger import configure_logging, get_logger
from ai_dev_agent.tools.execution.shell_session import ShellSessionError, ShellSessionManager

from .react.executor import _execute_react_assistant
from .utils import _build_context, get_llm_client, _record_invocation

LOGGER = get_logger(__name__)


class NaturalLanguageGroup(click.Group):
    """Group that falls back to NL intent routing when no command matches."""

    def resolve_command(self, ctx: click.Context, args: List[str]):  # type: ignore[override]
        planning_flag: Optional[bool] = None
        filtered_args: List[str] = []

        for arg in args:
            if arg == "--plan":
                planning_flag = True
            elif arg == "--direct":
                planning_flag = False
            else:
                filtered_args.append(arg)

        try:
            return super().resolve_command(ctx, filtered_args)
        except click.UsageError:
            if not filtered_args:
                raise
            if any(arg.startswith("-") for arg in filtered_args):
                raise
            query = " ".join(filtered_args).strip()
            if not query:
                raise
            ctx.meta["_pending_nl_prompt"] = query
            ctx.meta["_emit_status_messages"] = True
            if planning_flag is not None:
                ctx.meta["_use_planning"] = planning_flag
            return super().resolve_command(ctx, ["query"])


@click.group(cls=NaturalLanguageGroup)
@click.option("--config", "config_path", type=click.Path(path_type=Path), help="Path to config file.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging output.")
@click.option("--plan", is_flag=True, help="Use planning mode for all queries")
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None, verbose: bool, plan: bool) -> None:
    """AI-assisted development agent CLI."""
    from ai_dev_agent.cli import load_settings as _load_settings

    settings = _load_settings(config_path)
    if verbose:
        settings.log_level = "DEBUG"
    configure_logging(settings.log_level, structured=settings.structured_logging)
    if not settings.api_key:
        LOGGER.warning("No API key configured. Some commands may fail.")
    ctx.obj = _build_context(settings)
    ctx.obj["default_use_planning"] = plan


@cli.command(name="query")
@click.argument("prompt", nargs=-1)
@click.option("--plan", "force_plan", is_flag=True, help="Force planning for this query")
@click.option("--direct", is_flag=True, help="Force direct execution (no planning)")
@click.pass_context
def query(
    ctx: click.Context,
    prompt: Tuple[str, ...],
    force_plan: bool,
    direct: bool,
) -> None:
    """Execute a natural-language query using the ReAct workflow."""
    pending = " ".join(prompt).strip()
    if not pending:
        pending = str(ctx.meta.pop("_pending_nl_prompt", "")).strip()
    if not pending:
        pending = str(ctx.obj.pop("_pending_nl_prompt", "")).strip()
    if not pending:
        raise click.UsageError("Provide a request for the assistant.")

    _record_invocation(ctx, overrides={"prompt": pending, "mode": "query"})

    settings: Settings = ctx.obj["settings"]

    planning_pref = ctx.meta.pop("_use_planning", None)
    if planning_pref is None:
        planning_pref = ctx.obj.get("default_use_planning", False)

    use_planning = bool(planning_pref)
    if getattr(settings, "always_use_planning", False):
        use_planning = True

    if force_plan:
        use_planning = True
    elif direct:
        use_planning = False

    if not settings.api_key:
        raise click.ClickException(
            "No API key configured (DEVAGENT_API_KEY). Natural language assistance requires an LLM."
        )

    try:
        cli_pkg = import_module('ai_dev_agent.cli')
        llm_factory = getattr(cli_pkg, 'get_llm_client', get_llm_client)
    except ModuleNotFoundError:
        llm_factory = get_llm_client
    try:
        client = llm_factory(ctx)
    except click.ClickException as exc:
        raise click.ClickException(f'Failed to create LLM client: {exc}') from exc
    _execute_react_assistant(ctx, client, settings, pending, use_planning=use_planning)


@cli.command()
@click.pass_context
def shell(ctx: click.Context) -> None:
    """Start an interactive shell session with persistent context."""
    settings: Settings = ctx.obj["settings"]

    manager = ShellSessionManager(
        shell=getattr(settings, "shell_executable", None),
        default_timeout=getattr(settings, "shell_session_timeout", None),
        cpu_time_limit=getattr(settings, "shell_session_cpu_time_limit", None),
        memory_limit_mb=getattr(settings, "shell_session_memory_limit_mb", None),
    )

    try:
        session_id = manager.create_session(cwd=Path.cwd())
    except ShellSessionError as exc:
        raise click.ClickException(f"Failed to start shell session: {exc}") from exc

    previous_manager = ctx.obj.get("_shell_session_manager")
    previous_session = ctx.obj.get("_shell_session_id")
    previous_history = ctx.obj.get("_shell_conversation_history")
    ctx.obj["_shell_session_manager"] = manager
    ctx.obj["_shell_session_id"] = session_id
    ctx.obj["_shell_conversation_history"] = []

    click.echo("DevAgent Interactive Shell")
    click.echo("Type a question or command, 'help' for guidance, and 'exit' to quit.")
    click.echo("=" * 50)

    try:
        while True:
            try:
                user_input = click.prompt("DevAgent> ", prompt_suffix="", show_default=False).strip()
            except (KeyboardInterrupt, EOFError):
                click.echo("\nGoodbye!")
                break

            if not user_input:
                continue

            lowered = user_input.lower()
            if lowered in {"exit", "quit", "q"}:
                click.echo("Goodbye!")
                break

            if lowered == "help":
                click.echo("Enter any natural-language request to run `devagent query`.")
                click.echo("Use 'exit' to leave the shell.")
                continue

            try:
                ctx.invoke(query, prompt=(user_input,))
            except ShellSessionError as exc:
                click.echo(f"Shell session error: {exc}")
                break
            except TimeoutError as exc:
                click.echo(f"Command timed out: {exc}")
            except click.ClickException as exc:
                click.echo(f"Error: {exc}")
    finally:
        if previous_manager is not None:
            ctx.obj["_shell_session_manager"] = previous_manager
        else:
            ctx.obj.pop("_shell_session_manager", None)

        if previous_session is not None:
            ctx.obj["_shell_session_id"] = previous_session
        else:
            ctx.obj.pop("_shell_session_id", None)

        if previous_history is not None:
            ctx.obj["_shell_conversation_history"] = previous_history
        else:
            ctx.obj.pop("_shell_conversation_history", None)

        manager.close_all()


def main() -> None:
    cli(prog_name="devagent")


if __name__ == "__main__":  # pragma: no cover
    main()
