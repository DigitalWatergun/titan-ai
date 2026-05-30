import logging
import subprocess

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

BLOCKED_COMMANDS = [
    "rm -rf /",
    "rm -rf ~",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",
    "chmod -R 777 /",
    "grep -r",
    "find /",
]


class RunCommandInput(BaseModel):
    """Run a shell command and return stdout and stderr.
    Use this for running tests, checking git status, installing packages, etc."""

    command: str = Field(description="Shell command to execute")


def run_command(args: RunCommandInput) -> str:
    # Safety: block known destructive commands
    for blocked in BLOCKED_COMMANDS:
        if blocked in args.command:
            logger.warning(f"Blocked destructive command: {args.command}")
            return f"BLOCKED: '{args.command}' matches blocked pattern '{blocked}'. Refusing to execute."

    try:
        result = subprocess.run(
            args.command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=None,
        )
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        output += f"Return code: {result.returncode}"
        return output or "Command completed with no output"
    except subprocess.TimeoutExpired:
        logger.exception(f"Command timed out: {args.command}")
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        logger.exception(f"Failed to run command: {args.command}")
        return f"Error running command: {e}"
