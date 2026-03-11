"""
mail_helpers.py — Почтовый ящик, модели писем, тулкит.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict
from uuid import uuid4

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, tool
from langchain_core.tools.base import BaseToolkit

from spade_llm.core.conf import ConfigurableRecord, Configurable, configuration
from spade_llm.core.tools import ToolFactory

logger = logging.getLogger(__name__)


# ========================
# Модели данных
# ========================

class MailMessage(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:8], description="Unique mail ID")
    thread_id: str = Field(default="", description="Thread ID")
    sender: str = Field(description="Sender address")
    recipient: str = Field(description="Recipient address")
    subject: str = Field(default="", description="Subject")
    body: str = Field(default="", description="Body text")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="ISO timestamp")
    is_read: bool = Field(default=False, description="Read flag")


class MailThread(BaseModel):
    thread_id: str = Field(default_factory=lambda: uuid4().hex[:8], description="Thread ID")
    subject: str = Field(default="", description="Thread subject")
    messages: List[MailMessage] = Field(default_factory=list)

    def add_message(self, msg: MailMessage) -> None:
        msg.thread_id = self.thread_id
        self.messages.append(msg)


# ========================
# Mailbox (in-memory хранилище)
# ========================

class Mailbox:
    def __init__(self):
        self._messages: Dict[str, MailMessage] = {}
        self._threads: Dict[str, MailThread] = {}

    def clear(self):
        self._messages.clear()
        self._threads.clear()

    def add_message(self, msg: MailMessage) -> MailMessage:
        if not msg.thread_id:
            thread = MailThread(subject=msg.subject)
            self._threads[thread.thread_id] = thread
            msg.thread_id = thread.thread_id
        else:
            thread = self._threads.get(msg.thread_id)
            if thread is None:
                thread = MailThread(thread_id=msg.thread_id, subject=msg.subject)
                self._threads[thread.thread_id] = thread
        thread.add_message(msg)
        self._messages[msg.id] = msg
        logger.info("Mail %s added to thread %s", msg.id, msg.thread_id)
        return msg

    def get_message(self, mail_id: str) -> Optional[MailMessage]:
        return self._messages.get(mail_id)

    def delete_message(self, mail_id: str) -> bool:
        msg = self._messages.pop(mail_id, None)
        if msg is None:
            return False
        thread = self._threads.get(msg.thread_id)
        if thread:
            thread.messages = [m for m in thread.messages if m.id != mail_id]
            if not thread.messages:
                del self._threads[msg.thread_id]
        return True

    def list_messages(self, recipient: Optional[str] = None) -> List[MailMessage]:
        msgs = list(self._messages.values())
        if recipient:
            msgs = [m for m in msgs if m.recipient == recipient]
        return sorted(msgs, key=lambda m: m.timestamp, reverse=True)

    def get_thread(self, thread_id: str) -> Optional[MailThread]:
        return self._threads.get(thread_id)

    def list_threads(self) -> List[MailThread]:
        return list(self._threads.values())


# ========================
# Глобальный shared mailbox
# ========================

_SHARED_MAILBOX: Optional[Mailbox] = None


def get_shared_mailbox() -> Mailbox:
    """Единый Mailbox, доступный и тулкиту, и агентам."""
    global _SHARED_MAILBOX
    if _SHARED_MAILBOX is None:
        _SHARED_MAILBOX = Mailbox()
    return _SHARED_MAILBOX


# ========================
# Тулкит
# ========================

class MailToolkit(BaseToolkit):
    mailbox: Mailbox = Field(description="Mailbox instance")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, mailbox: Mailbox):
        super().__init__(mailbox=mailbox)

    def get_tools(self) -> List[BaseTool]:
        return [
            self._create_list_mails_tool(),
            self._create_get_mail_tool(),
            self._create_delete_mail_tool(),
        ]

    def _create_list_mails_tool(self) -> BaseTool:
        mailbox = self.mailbox

        @tool
        def list_mails(recipient: str = "") -> str:
            """
            Return a list of all mails in the mailbox.

            Arguments:
                recipient: (optional) filter by recipient address. Empty string = all mails.

            Returns:
                Formatted list of mails with id, sender, subject, timestamp, read status.
            """
            msgs = mailbox.list_messages(recipient=recipient or None)
            if not msgs:
                return "Mailbox is empty."
            lines = []
            for m in msgs:
                status = "read" if m.is_read else "unread"
                lines.append(
                    f"[{m.id}] from={m.sender} subject=\"{m.subject}\" "
                    f"time={m.timestamp} thread={m.thread_id} status={status}"
                )
            return "\n".join(lines)

        return list_mails

    def _create_get_mail_tool(self) -> BaseTool:
        mailbox = self.mailbox

        @tool
        def get_mail(mail_id: str) -> str:
            """
            Get full content of a mail by its ID. Marks it as read.

            Arguments:
                mail_id: The unique identifier of the mail.

            Returns:
                Full mail content or error message.
            """
            msg = mailbox.get_message(mail_id)
            if msg is None:
                return f"Mail '{mail_id}' not found."
            msg.is_read = True
            return (
                f"ID: {msg.id}\n"
                f"Thread: {msg.thread_id}\n"
                f"From: {msg.sender}\n"
                f"To: {msg.recipient}\n"
                f"Subject: {msg.subject}\n"
                f"Date: {msg.timestamp}\n"
                f"---\n"
                f"{msg.body}"
            )

        return get_mail

    def _create_delete_mail_tool(self) -> BaseTool:
        mailbox = self.mailbox

        @tool
        def delete_mail(mail_id: str) -> str:
            """
            Delete a mail by its ID.

            Arguments:
                mail_id: The unique identifier of the mail to delete.

            Returns:
                Confirmation or error message.
            """
            ok = mailbox.delete_message(mail_id)
            if ok:
                return f"Mail '{mail_id}' deleted successfully."
            return f"Mail '{mail_id}' not found."

        return delete_mail


# ========================
# ToolFactory для spade-llm
# ========================

class MailToolkitConf(ConfigurableRecord):
    pass


@configuration(MailToolkitConf)
class MailToolkitFactory(ToolFactory, Configurable[MailToolkitConf]):
    def create_tool(self) -> List[BaseTool]:
        return MailToolkit(mailbox=get_shared_mailbox()).get_tools()