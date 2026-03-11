"""
scenario.py — Модели сценария для почтового бенчмарка.
"""

import logging
import yaml
from typing import List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ScenarioEmail(BaseModel):
    """Письмо в сценарии."""
    id: str = Field(description="Unique email ID within scenario")
    sender: str = Field(description="Sender address")
    recipient: str = Field(default="user@company.com", description="Recipient address")
    subject: str = Field(default="", description="Subject line")
    body: str = Field(default="", description="Email body text")
    reply_to: Optional[str] = Field(default=None, description="ID of email this is a reply to (same thread)")
    delay: float = Field(default=0.0, description="Delay in seconds before email appears in mailbox")


class ScenarioQuestion(BaseModel):
    """Вопрос для проверки агента."""
    question: str = Field(description="Question text to ask the mail agent")
    expected_answer: str = Field(description="Key fact or substring expected in the answer")
    ask_after: float = Field(default=0.0, description="Seconds to wait before asking this question")


class Scenario(BaseModel):
    """Полный сценарий: письма + вопросы."""
    name: str = Field(default="default", description="Scenario name")
    emails: List[ScenarioEmail] = Field(default_factory=list)
    questions: List[ScenarioQuestion] = Field(default_factory=list)


def load_scenario(path: str) -> Scenario:
    """Загрузить сценарий из YAML-файла."""
    logger.info("Loading scenario from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Scenario(**data)