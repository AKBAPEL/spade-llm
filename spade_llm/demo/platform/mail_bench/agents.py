"""
agents.py — EnvironmentAgent (загрузка сценариев, оценка) и MailAgent (LLM + tools).
"""

import asyncio
import logging
from typing import List, Dict, Optional

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

try:
    from mem0 import Memory

    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    print("Mem0 is not available")
    Memory = None
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
        async def _notify_mail_agent(self, email_id: str):
            try:
                await self.context.inform("mail_agent").with_content(email_id)
                logger.debug("Notification sent for email %s", email_id)
            except Exception as e:
                logger.warning("Failed to notify mail_agent: %s", e)

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
                asyncio.create_task(self._notify_mail_agent(se.id))
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
                await self._notify_mail_agent(se.id)
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
                print('1111111111',sq.question[:20])
                # Ждать ответ
                response = await self.receive(
                    MessageTemplate(performative=consts.INFORM, thread_id=self.context.thread_id),
                    timeout=60
                )
                print('22222222222', response.content[:20])
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
                f"\n{'=' * 60}",
                f"  BENCHMARK RESULTS: {scenario.name}",
                f"{'=' * 60}",
            ]
            for i, r in enumerate(results, 1):
                mark = "✅" if r["correct"] else "❌"
                report_lines.append(f"  {mark} Q{i}: {r['question']}")
                report_lines.append(f"       Expected: {r['expected']}")
                report_lines.append(f"       Got:      {r['actual'][:120]}")
                report_lines.append("")

            report_lines.append(f"  SCORE: {passed}/{total} ({pct:.0f}%)")
            report_lines.append(f"{'=' * 60}\n")

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
    # Опции памяти Mem0
    use_memory: bool = Field(default=False, description="Enable Mem0 long-term memory")
    memory_config: Optional[dict] = Field(
        default=None,
        description="Configuration dict for Mem0 (if None and use_memory=True, uses default)"
    )


class MailRequestBehaviour(MessageHandlingBehavior):

    def __init__(self, system_prompt: str,
                 model: BaseChatModel,
                 max_iterations: int,
                 memory: Optional[Memory] = None,
                 user_id: str = "mail_agent_user",
                 email_store: Optional[List] = None):
        super().__init__(MessageTemplate.request())
        self.max_iterations = max_iterations
        self.model = model
        self.base_system_prompt = system_prompt
        self.memory = memory
        self.user_id = user_id
        self.email_store = email_store
    async def _ainvoke_with_retry(self, model, messages, max_retries=5, base_delay=1):
        """Вызов model.ainvoke с повторными попытками при ошибках 429 и других временных сбоях."""
        for attempt in range(max_retries):
            try:
                return await model.ainvoke(messages)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise  # последняя попытка – пробрасываем исключение
                error_str = str(e).lower()
                # Проверяем наличие признаков rate limit (429)
                if "429" in error_str or "too many requests" in error_str:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"Rate limited (429), retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                else:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"Temporary error: {e}, retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(delay)
        raise RuntimeError("Max retries exceeded without successful response")

    async def step(self):
        if not self.message:
            return

        msg = self.message
        self.logger.debug("Handling message %s", msg)

        tools = self.context.get_tools(self.agent)
        model = self.model.bind_tools(tools)
        tools_dict = {t.name: t for t in tools}

        # Подготовка системного промпта с учётом памяти (если включена)
        system_prompt = self.base_system_prompt
        if self.memory:
            try:
                # Поиск релевантных воспоминаний
                memories = self.memory.search(
                    query=msg.content,
                    user_id=self.user_id,
                    limit=3
                )
                memories_text = "\n".join(
                    f"- {m['memory']}" for m in memories.get("results", [])
                )
                self.logger.info("Found relevant fragments: %s", memories_text)
                if memories_text:
                    system_prompt += f"\n\nRelevant memories:\n{memories_text}"
                    self.logger.debug("Added %d memories to prompt", len(memories["results"]))
            except Exception as e:
                self.logger.warning("Failed to retrieve memories: %s", e)
        elif self.email_store is not None:
            # Если память отключена, используем локальное хранилище писем
            if self.email_store:
                emails_text = "\n".join(self.email_store)
                system_prompt += f"\n\nReceived emails:\n{emails_text}"
                self.logger.debug("Added %d emails to prompt", len(self.email_store))
        # Начальные сообщения (системное + пользовательское)
        current_messages = [SystemMessage(system_prompt)]
        user_message = HumanMessage(msg.content)
        history = []  # будет содержать всё, что после user_message

        for iteration in range(self.max_iterations):
            answer = await self._ainvoke_with_retry(
                model, current_messages + [user_message] + history
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
                # Финальный ответ без вызова инструментов
                reply_content = answer.content
                await self.context.reply_with_inform(msg).with_content(reply_content)
                return

        # Исчерпаны итерации
        await self.context.reply_with_failure(msg).with_content("Max iterations reached")


class MailReceiveBehaviour(MessageHandlingBehavior):
    def __init__(self, memory: Optional[Memory] = None,
                 user_id: str = "mail_agent_user",
                 email_store: Optional[List] = None):
        super().__init__(MessageTemplate.inform())
        self.memory = memory
        self.user_id = user_id
        self.received_email_ids: List[str] = []  # локальное хранилище при отключённой памяти
        self.email_store = email_store  # сохраняем

    async def step(self):
        if not self.message:
            return

        msg = self.message
        email_id = msg.content.strip()
        self.logger.debug("Received notification for email %s", email_id)

        # Получаем письмо из общего mailbox
        mailbox = get_shared_mailbox()
        email = mailbox.get_message(email_id)
        if not email:
            self.logger.warning("Email %s not found in mailbox", email_id)
            return

        if self.memory:
            # Сохраняем факт о письме в Mem0
            try:
                fact = f"Письмо от {email.sender} на тему '{email.subject}': {email.body}"
                messages_for_memory = [
                    {"role": "user", "content": f"Новое письмо: {email.subject}"},
                    {"role": "assistant", "content": fact}
                ]
                # Добавляем метаданные для возможной фильтрации
                metadata = {
                    "email_id": email_id,
                    "sender": email.sender,
                    "subject": email.subject,
                    "thread_id": email.thread_id
                }
                self.memory.add(messages_for_memory, user_id=self.user_id, metadata=metadata)
                self.logger.debug("Saved fact about email %s to memory", email_id)
            except Exception as e:
                self.logger.warning("Failed to save email fact to memory: %s", e)
        else:
            if self.email_store is not None:
                # Формируем текстовое представление письма
                email_text = f"От: {email.sender}\nТема: {email.subject}\nТекст: {email.body}"
                self.email_store.append(email_text)
                self.logger.debug("Stored email %s locally", email_id)


@configuration(MailAgentConf)
class MailAgent(Agent, Configurable[MailAgentConf]):
    def setup(self):
        user_id = "mail_agent_user"
        SYSTEM_PROMPT = (
            "Ты персональный почтовый помощник. "
            "Используй предоставленные инструменты для работы с почтой. "
            "Сначала получи список писем, затем при необходимости прочитай нужные письма. "
            "Отвечай на вопросы точно и кратко, опираясь на содержимое писем."
        )
        memory = None
        email_store = None
        if self.config.use_memory:
            # Инициализация памяти Mem0, если запрошено
            if self.config.memory_config is None:
                # ВАЖНО: подставьте правильную размерность эмбеддингов!
                EMBEDDING_DIM = 2560  # уточните под вашу модель
                self.config.memory_config = {
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": self.config.model,
                            "openai_base_url": "http://localhost:8090/v1",  # или из окружения
                            "api_key": "dummy"
                        }
                    },
                    "embedder": {
                        "provider": "openai",
                        "config": {
                            "model": "EmbeddingsGigaR",  # уточните название
                            "openai_base_url": "http://localhost:8090/v1",
                            "api_key": "dummy",
                            "embedding_dims": EMBEDDING_DIM
                        }
                    },
                    "vector_store": {
                        "provider": "chroma",
                        "config": {
                            "collection_name": "spade_test_mem0_chroma_v2",
                            "path": "/tmp/chroma_mem0"
                            # поле "dimension" удалено — оно не допускается в vector_store.config
                        }
                    }
                }
                memory = Memory.from_config(self.config.memory_config)
                logger.info("Mem0 memory initialized")
        else:
            # Создаём локальное хранилище писем
            email_store = []
            logger.info("Using local email store (memory disabled)")

        self.add_behaviour(MailRequestBehaviour(
            system_prompt=SYSTEM_PROMPT,
            model=self.default_context.create_chat_model(self.config.model),
            max_iterations=self.config.max_iterations,
            memory=memory,  # добавляем
            user_id=user_id,
            email_store=email_store  # добавляем
        ))
        self.add_behaviour(MailReceiveBehaviour(
            memory=memory,
            user_id=user_id,
            email_store=email_store
        ))
