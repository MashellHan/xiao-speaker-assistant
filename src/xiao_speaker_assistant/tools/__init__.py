"""Tool integrations (Outlook, Xiaohongshu MCP, Edge Browser, and scoped macOS file helpers)."""

from .browser import (
    EDGE_APP_NAME,
    build_edge_open_command,
    extract_html_summary,
    fetch_webpage_text,
    open_in_edge,
    truncate_text,
    validate_http_url,
)
from .confirmation import ConfirmationStore, is_cancel_utterance, is_confirm_utterance
from .mac_local import (
    DEFAULT_ALLOWED_ROOTS,
    MacLocalAccessError,
    is_sensitive_path,
    list_recent_files,
    read_local_text_file,
    resolve_allowed_roots,
)
from .mcp_client import HttpMcpClient
from .mcp_server import ALL_TOOL_DEFINITIONS, LOCAL_TOOL_DEFINITIONS, TOOL_DEFINITIONS, McpServer
from .memory import MemoryEntry, MemorySecurityError, MemoryStore
from .outlook import OutlookClient, PendingEmailDraft
from .reminders import ReminderEntry, ReminderScheduler, ReminderStore, is_quiet_hours

__all__ = [
    "ALL_TOOL_DEFINITIONS",
    "ConfirmationStore",
    "DEFAULT_ALLOWED_ROOTS",
    "EDGE_APP_NAME",
    "HttpMcpClient",
    "LOCAL_TOOL_DEFINITIONS",
    "MacLocalAccessError",
    "McpServer",
    "MemoryEntry",
    "MemorySecurityError",
    "MemoryStore",
    "OutlookClient",
    "PendingEmailDraft",
    "ReminderEntry",
    "ReminderScheduler",
    "ReminderStore",
    "is_quiet_hours",
    "TOOL_DEFINITIONS",
    "build_edge_open_command",
    "extract_html_summary",
    "fetch_webpage_text",
    "is_cancel_utterance",
    "is_confirm_utterance",
    "is_sensitive_path",
    "list_recent_files",
    "open_in_edge",
    "read_local_text_file",
    "resolve_allowed_roots",
    "truncate_text",
    "validate_http_url",
]
