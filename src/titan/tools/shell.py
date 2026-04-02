import logging
import subprocess

from langchain_core.tools import tool
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
    command: str = Field(description="Shell command to execute")


@tool(args_schema=RunCommandInput)
def run_command(command: str) -> str:
    """Run a shell command and return stdout and stderr.
    Use this for running tests, checking git status, installing packages. etc."""
    # Safety: block known destructive commands
    for blocked in BLOCKED_COMMANDS:
        if blocked in command:
            logger.warning(f"Blocked destructive command: {command}")
            return f"BLOCKED: '{command}' matches blocked pattern '{blocked}'. Refusing to execute."

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30, cwd=None
        )
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        output += f"Return code: {result.returncode}"
        return output or "Command complated with no output"
    except subprocess.TimeoutExpired:
        logger.exception(f"Command timed out: {command}")
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        logger.exception(f"Failed to run command: {command}")
        return f"Error running command: {e}"
