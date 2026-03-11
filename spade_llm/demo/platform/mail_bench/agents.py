"""
agents.py — EnvironmentAgent (загрузка сценариев, оценка) и MailAgent (LLM + tools).
"""

import asyncio
import logging
from typing import List, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field

from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate
from spade_llm.core.conf import Configurable, configuration
from spade_llm.core.api import Message
from spade_llm import consts

from spade_llm.demo.platform.mail_bench.mail_helpers import (
    get_shared_mailbox, MailMessage, Mailbox
)
from spade_llm.demo.platform.mail_bench.scenario import (
    load_scenario, Scenario, ScenarioEmail
)

logger = logging.getLogger(__name__)


# =====================================================
# EnvironmentAgent — загружает сценарий, гоняет бенчмарк
# =====================================================

class EnvironmentAgentConf(BaseModel):
    scenario: str = Field(description="Path to the scenario YAML file")
    model: str = Field(description="Model name for answer evaluation")


@configuration(EnvironmentAgentConf)
class EnvironmentAgent(Agent, Configurable[EnvironmentAgentConf]):

    class RunScenarioBehaviour(MessageHandlingBehavior):

        def __init__(self, config: EnvironmentAgentConf, eval_model: BaseChatModel):
            super().__init__(MessageTemplate.request())
            self.config = config
            self.eval_model = eval_model

        # ---------- вспомогательное ----------

        def _populate_immediate(self, scenario: Scenario, mailbox: Mailbox) -> Dict[str, str]:
            """
            Добавить в mailbox письма с delay=0.
            Возвращает маппинг scenario_email_id -> реальный thread_id
            (нужен для reply_to).
            """
            thread_map: Dict[str, str] = {}  # scenario email id -> thread_id

            immediate = [e for e in scenario.emails if e.delay <= 0]
            for se in immediate:
                msg = self._scenario_email_to_mail(se, thread_map)
                mailbox.add_message(msg)
                thread_map[se.id] = msg.thread_id
            return thread_map

        async def _schedule_delayed(self, scenario: Scenario, mailbox: Mailbox,
                                    thread_map: Dict[str, str]):
            """Фоновая задача: добавляет письма с delay > 0 по таймеру."""
            delayed = sorted(
                [e for e in scenario.emails if e.delay > 0],
                key=lambda e: e.delay
            )
            elapsed = 0.0
            for se in delayed:
                wait = se.delay - elapsed
                if wait > 0:
                    await asyncio.sleep(wait)
                    elapsed = se.delay
                msg = self._scenario_email_to_mail(se, thread_map)
                mailbox.add_message(msg)
                thread_map[se.id] = msg.thread_id
                logger.info("Delayed email '%s' delivered after %.1fs", se.id, se.delay)

        @staticmethod
        def _scenario_email_to_mail(se: ScenarioEmail, thread_map: Dict[str, str]) -> MailMessage:
            """Конвертировать ScenarioEmail -> MailMessage, разрешая reply_to -> thread_id."""
            thread_id = ""
            if se.reply_to and se.reply_to in thread_map:
                thread_id = thread_map[se.reply_to]

            return MailMessage(
                id=se.id,
                sender=se.sender,
                recipient=se.recipient,
                subject=se.subject,
                body=se.body,
                thread_id=thread_id,
            )

        async def _evaluate_answer(self, question: str, expected: str, actual: str) -> bool:
            """Используем LLM-судью: содержит ли ответ ожидаемый факт?"""
            prompt = (
                "Ты строгий судья. Определи, содержит ли ОТВЕТ ключевой факт из ОЖИДАНИЯ.\n"
                "Отвечай ОДНИМ словом: ДА или НЕТ.\n\n"
                f"ВОПРОС: {question}\n"
                f"ОЖИДАНИЕ: {expected}\n"
                f"ОТВЕТ: {actual}"
            )
            result = await self.eval_model.ainvoke([HumanMessage(prompt)])
            verdict = result.content.strip().upper()
            logger.debug("Evaluation: expected='%s' actual='%s' verdict='%s'", expected, actual, verdict)
            return "ДА" in verdict

        # ---------- основной цикл ----------

        async def step(self) -> None:
            if not self.message:
                return

            trigger_msg = self.message

            # 1. Загрузить сценарий
            scenario = load_scenario(self.config.scenario)
            logger.info("Loaded scenario '%s': %d emails, %d questions",
                        scenario.name, len(scenario.emails), len(scenario.questions))

            # 2. Подготовить mailbox
            mailbox = get_shared_mailbox()
            mailbox.clear()

            # 3. Добавить немедленные письма
            thread_map = self._populate_immediate(scenario, mailbox)

            # 4. Запустить доставку отложенных писем в фоне
            delayed_task = asyncio.create_task(
                self._schedule_delayed(scenario, mailbox, thread_map)
            )

            # 5. Задавать вопросы и собирать ответы
            results: List[dict] = []

            for sq in scenario.questions:
                # Подождать если нужно (чтобы письма успели "дойти")
                if sq.ask_after > 0:
                    logger.info("Waiting %.1fs before asking next question...", sq.ask_after)
                    await asyncio.sleep(sq.ask_after)

                # Отправить вопрос mail_agent-у
                await self.context.request("mail_agent").with_content(sq.question)

                # Ждать ответ
                response = await self.receive(
                    MessageTemplate(performative=consts.INFORM, thread_id=self.context.thread_id),
                    timeout=60
                )

                actual_answer = response.content if response else "<NO RESPONSE>"

                # Оценить
                correct = await self._evaluate_answer(sq.question, sq.expected_answer, actual_answer)

                results.append({
                    "question": sq.question,
                    "expected": sq.expected_answer,
                    "actual": actual_answer,
                    "correct": correct,
                })

                status = "✅" if correct else "❌"
                logger.info("%s Q: %s | Expected: %s | Actual: %s | Correct: %s",
                            status, sq.question, sq.expected_answer, actual_answer, correct)

            # 6. Дождаться завершения доставки
            delayed_task.cancel()

            # 7. Сформировать сводку
            total = len(results)
            passed = sum(1 for r in results if r["correct"])
            pct = (passed / total * 100) if total > 0 else 0

            report_lines = [
                f"\n{'='*60}",
                f"  BENCHMARK RESULTS: {scenario.name}",
                f"{'='*60}",
            ]
            for i, r in enumerate(results, 1):
                mark = "✅" if r["correct"] else "❌"
                report_lines.append(f"  {mark} Q{i}: {r['question']}")
                report_lines.append(f"       Expected: {r['expected']}")
                report_lines.append(f"       Got:      {r['actual'][:120]}")
                report_lines.append("")

            report_lines.append(f"  SCORE: {passed}/{total} ({pct:.0f}%)")
            report_lines.append(f"{'='*60}\n")

            report = "\n".join(report_lines)
            print(report)

            # Вернуть результат в консоль
            await self.context.reply_with_inform(trigger_msg).with_content(
                f"Benchmark done. Score: {passed}/{total} ({pct:.0f}%)"
            )

    def setup(self):
        eval_model = self.default_context.create_chat_model(self.config.model)
        self.add_behaviour(self.RunScenarioBehaviour(self.config, eval_model))


# =====================================================
# MailAgent — LLM + tools
# =====================================================

class MailAgentConf(BaseModel):
    model: str = Field(description="Model to use for handling messages")
    max_iterations: int = Field(default=10, description="Max tool-call iterations per message")


class MailRequestBehaviour(MessageHandlingBehavior):

    def __init__(self, template: MessageTemplate, system_prompt: str,
                 model: BaseChatModel, max_iterations: int):
        super().__init__(template)
        self.max_iterations = max_iterations
        self.model = model
        self.initial_messages = [SystemMessage(system_prompt)]

    async def step(self):
        if not self.message:
            return

        msg = self.message
        self.logger.debug("Handling message %s", msg)

        tools = self.context.get_tools(self.agent)
        model = self.model.bind_tools(tools)
        tools_dict = {t.name: t for t in tools}

        history = []
        user_message = HumanMessage(msg.content)

        for _ in range(self.max_iterations):
            answer = await model.ainvoke(
                self.initial_messages + [user_message] + history
            )
            self.logger.debug("Got answer %s", answer)
            history.append(answer)

            if isinstance(answer, AIMessage) and answer.tool_calls:
                for tc in answer.tool_calls:
                    name = tc["name"]
                    self.logger.debug("Invoking tool %s", name)
                    if name in tools_dict:
                        result = await tools_dict[name].ainvoke(tc["args"])
                        self.logger.debug("Tool %s returned %s", name, result)
                        history.append(ToolMessage(result, tool_call_id=tc["id"]))
                    else:
                        self.logger.warning("Tool %s not found", name)
                        await self.context.reply_with_failure(msg).with_content(
                            f"Tool not found: {name}"
                        )
                        return
            else:
                await self.context.reply_with_inform(msg).with_content(answer.content)
                return

        # Исчерпаны итерации
        await self.context.reply_with_failure(msg).with_content("Max iterations reached")


@configuration(MailAgentConf)
class MailAgent(Agent, Configurable[MailAgentConf]):
    def setup(self):
        SYSTEM_PROMPT = (
            "Ты персональный почтовый помощник. "
            "Используй предоставленные инструменты для работы с почтой. "
            "Сначала получи список писем, затем при необходимости прочитай нужные письма. "
            "Отвечай на вопросы точно и кратко, опираясь на содержимое писем."
        )
        self.add_behaviour(MailRequestBehaviour(
            template=MessageTemplate.request(),
            system_prompt=SYSTEM_PROMPT,
            model=self.default_context.create_chat_model(self.config.model),
            max_iterations=self.config.max_iterations,
        ))