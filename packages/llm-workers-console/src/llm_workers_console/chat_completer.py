from logging import getLogger
from pathlib import Path
from typing import List

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style

logger = getLogger(__name__)


class ChatCompleter(Completer):
    """Completer for slash commands and @-triggered file paths in chat interface."""

    # Minimal style: only the active completion is bold.
    style = Style.from_dict({
        "completion-menu.completion.current": "bold",
        # (Optional) keep the rest untouched; no colors set.
        "completion-menu.completion": "",
        "completion-menu.meta.completion": "",
        "completion-menu.meta.completion.current": "",
        "scrollbar.background": "",
        "scrollbar.button": "",
    })

    def __init__(self, commands_config: dict):
        """Initialize completer with available commands and formatting.

        Args:
            commands_config: Dictionary of command configurations
        """
        self.commands_config = commands_config

        # Calculate the maximum width for command column alignment
        command_strs = []
        for cmd, config in commands_config.items():
            cmd_str, _ = self._format_command_display_raw(cmd, config)
            command_strs.append(cmd_str)

        self.max_cmd_width = max(len(cmd_str) for cmd_str in command_strs) if command_strs else 0

    @staticmethod
    def _format_command_display_raw(cmd: str, config: dict) -> tuple[str, str]:
        """Format a command for display with aliases and parameters (without alignment).

        Args:
            cmd: Primary command name
            config: Command configuration dict

        Returns:
            Tuple of (formatted_command_string, description)
        """
        # Build command string with aliases and params
        cmd_aliases = [cmd]
        if "aliases" in config:
            cmd_aliases.extend(config["aliases"])

        cmd_str = "/" + ", /".join(cmd_aliases)
        if "params" in config:
            cmd_str += f" {config['params']}"

        return cmd_str, config["description"]

    def _format_command_display(self, cmd: str, config: dict) -> tuple[str, str]:
        """Format a command for display with full alignment.

        Args:
            cmd: Primary command name
            config: Command configuration dict

        Returns:
            Tuple of (primary_command, fully_formatted_aligned_string)
        """
        cmd_str, description = self._format_command_display_raw(cmd, config)
        aligned_display = f"  {cmd_str:<{self.max_cmd_width}}  {description}"
        return cmd, aligned_display

    @staticmethod
    def _find_matching_files(prefix: str) -> List[str]:
        """Find files matching the given prefix in current directory and subdirectories.

        Args:
            prefix: The prefix to match against filenames (case-insensitive)

        Returns:
            List of relative file paths matching the prefix, limited to 20 entries
        """
        matches = []
        prefix_lower = prefix.lower()

        try:
            # Use pathlib to walk through current directory and subdirectories
            current_path = Path(".")
            for file_path in current_path.rglob("*"):
                # Skip directories, only include files
                if file_path.is_file():
                    relative_path = str(file_path)
                    filename = file_path.name

                    # Check if filename starts with prefix (case-insensitive)
                    if filename.lower().startswith(prefix_lower):
                        matches.append(relative_path)

                        # Limit to 20 entries as specified
                        if len(matches) >= 20:
                            break

            # Sort matches alphabetically for consistent ordering
            matches.sort()

        except Exception as e:
            logger.warning(f"Error finding files for prefix '{prefix}': {e}")

        return matches

    def _get_file_completions(self, document, prefix: str):
        """Generate file completions for @-triggered file paths.

        Args:
            document: The prompt-toolkit Document object
            prefix: The prefix after '@' to match against

        Yields:
            Completion objects for matching files
        """
        matches = self._find_matching_files(prefix)

        for file_path in matches:
            # Quote the file path to handle spaces and special characters
            quoted_path = f'"{file_path}"'

            yield Completion(
                text=quoted_path,  # insert quoted file path (without '@')
                start_position=-(len(prefix) + 1),  # replace '@prefix'
                display=f"  {file_path}",  # show clean relative path
            )

    @staticmethod
    def _find_at_word_before_cursor(text_before_cursor: str) -> str | None:
        """Find if there's an @-prefixed word immediately before the cursor.

        Args:
            text_before_cursor: The text before the cursor position

        Returns:
            The @-prefixed word if found and valid, None otherwise
        """
        if not text_before_cursor:
            return None

        # Split by spaces and analyze the last segment
        parts = text_before_cursor.split(' ')
        last_part = parts[-1] if parts else ""

        # Check conditions:
        # 1. Must start with '@'
        # 2. Must be more than 3 chars total (including @)
        # 3. Must not end with special chars that would indicate it's not a filename
        if (last_part.startswith('@') and
            len(last_part) > 3):
            return last_part

        return None

    def get_completions(self, document, complete_event):
        """Generate completions for slash commands and @-triggered file paths."""
        text = document.text_before_cursor

        # Check for @-triggered file completion first (can appear anywhere in text)
        at_word = self._find_at_word_before_cursor(text)
        if at_word:
            prefix = at_word[1:]  # part after '@'
            yield from self._get_file_completions(document, prefix)

        # Handle slash commands only if text starts with "/" and we haven't found @ completion
        elif text.startswith("/"):
            prefix = text[1:]  # part after "/"

            for cmd, config in self.commands_config.items():
                # Check if primary command or any alias matches the prefix
                all_names = [cmd]
                if "aliases" in config:
                    all_names.extend(config["aliases"])

                if any(name.startswith(prefix) for name in all_names):
                    # Use formatting logic
                    _, aligned_display = self._format_command_display(cmd, config)

                    yield Completion(
                        text=cmd,                     # insert only primary command name
                        start_position=-len(prefix),  # replace just the typed part (after /)
                        display=aligned_display,      # show full formatted display
                    )