from typing import Optional, Dict
from uuid import UUID

from langchain_core.messages import AIMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.style import Style
from rich.live import Live

from llm_workers.api import WorkerNotification, CONFIDENTIAL
from llm_workers.config import DisplaySettings
from llm_workers.worker import Worker


class ConsoleStream:
    """Handles visual cues and streaming output to the console using Rich library."""

    def __init__(self, console: Console, display_settings: DisplaySettings):
        self._console = console
        self._display_settings = display_settings
        self._reasoning_style = Style(italic=True, dim=True)
        self._streamed_message_id: Optional[str] = None
        self._streamed_reasoning_index: Optional[int] = None
        self._running_tools_depths: Dict[UUID, int] = {}
        self._thinking_live = None
        self._markdown_text: Optional[str] = None
        self._markdown_live: Optional[Live] = None

    def show_thinking(self):
        """Display 'Thinking...' message using Rich Live display."""
        if not self._thinking_live:
            self._thinking_live = self._console.status("Thinking...", spinner="dots")
            self._thinking_live.start()

    def clear_thinking_message(self):
        """Clear the 'Thinking...' message."""
        if self._thinking_live:
            self._thinking_live.stop()
            self._thinking_live = None

    def process_output_chunk(self, message_id: Optional[str], text: str):
        self.clear_thinking_message()
        if self._streamed_message_id is not None and self._streamed_message_id != message_id:
            self.clear()
        self._streamed_message_id = message_id
        self._streamed_reasoning_index = None
        if self._display_settings.markdown_output:
            if self._markdown_live is None:
                self._markdown_text = text
                self._markdown_live = Live(Markdown(text, code_theme="fruity"), console=self._console,
                                           refresh_per_second=1, vertical_overflow="visible", screen=False)
                self._markdown_live.start()
            else:
                self._markdown_text += text
                self._markdown_live.update(Markdown(self._markdown_text, code_theme="fruity"))
                # self._markdown_live.refresh()
        else:
            print(text, end="", flush=True)

    def process_reasoning_chunk(self, message_id: Optional[str], text: str, index: int):
        if not self._display_settings.show_reasoning:
            return
        self._clear(clear_message=(self._streamed_message_id != message_id or self._streamed_reasoning_index != index))
        if self._streamed_reasoning_index is None:
            self._console.print("Reasoning:")
            self._streamed_reasoning_index = index
        elif self._streamed_reasoning_index and index != self._streamed_reasoning_index:
            print()
            self._streamed_reasoning_index = index
        self._console.print(text, end="", style=self._reasoning_style)
        self._console.file.flush()

    def process_tool_start_notification(self, message: str, run_id: UUID, parent_run_id: Optional[UUID]):
        """Process tool_start notification."""
        self._clear(clear_tools=False)

        if parent_run_id is not None and parent_run_id in self._running_tools_depths:
            # increase depth of running tool
            depth = self._running_tools_depths[parent_run_id] + 1
            self._running_tools_depths[run_id] = depth
            ident = "  " * depth
            self._console.print(f"{ident}└ {message}...")
        else:
            self._running_tools_depths[run_id] = 0
            self._console.print(f"⏺ {message}...")

    def process_model_message(self, message: AIMessage):
        last_streamed_message_id = self._streamed_message_id
        self._clear()
        if last_streamed_message_id is not None and last_streamed_message_id == message.id:
            return
        if self._display_settings.show_reasoning:
            reasoning: list[WorkerNotification] = [
                notification
                for notification in Worker.extract_notifications(message_id=message.id, index=0, content=message.content)
                if notification.type == 'ai_reasoning_chunk'
            ]
            if len(reasoning) > 0:
                for index, notification in enumerate(reasoning):
                    self.process_reasoning_chunk(message.id, notification.text, index)
                self._clear()
        # text
        confidential = getattr(message, CONFIDENTIAL, False)
        if confidential:
            self._console.print("[Message below is confidential, not shown to AI Assistant]", style="bold red")
        if self._display_settings.markdown_output:
            self._console.print(Markdown(message.text()))
        else:
            self._console.print(message.text())
        if confidential:
            self._console.print("[Message above is confidential, not shown to AI Assistant]", style="bold red")

    def clear(self):
        """Finish any ongoing streaming output and clears internals state."""
        self._clear()

    def _clear(self, clear_thinking: bool = True, clear_message: bool = True, clear_tools: bool = True):
        """Internal method to clear state without printing newlines."""
        if clear_thinking:
            self.clear_thinking_message()

        if clear_message:
            if self._streamed_message_id:
                print()
                self._streamed_message_id = None
                self._streamed_reasoning_index = None
            if self._markdown_live:
                self._markdown_live.stop()
                self._markdown_live = None
                self._markdown_text = None

        if clear_tools:
            self._running_tools_depths.clear()