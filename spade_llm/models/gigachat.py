import asyncio
import logging
import time
from typing import List, Optional

from gigachat.exceptions import ResponseError
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_gigachat import GigaChatEmbeddings
from langchain_gigachat.chat_models import GigaChat

from spade_llm.core.conf import configuration, Configurable
from spade_llm.core.models import ChatModelFactory, EmbeddingsModelFactory, CredentialsUtils

SENSITIVE_KEYS = ["access_token", "password", "key_file_password", "credentials", "scope"]

logger = logging.getLogger(__name__)


class GigaChatWithExtra(GigaChat, extra="ignore"):
    pass


class GigaChatWithRetries(GigaChatWithExtra):
    _MAX_RETRIES: int = 3
    _RETRY_DELAYS: list[float] = [2.0, 4.0, 6.0]

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs,
    ) -> ChatResult:
        last_exc = None
        for attempt in range(self._MAX_RETRIES + 1):
            if attempt > 0:
                delay = self._RETRY_DELAYS[attempt - 1]
                logger.warning(
                    "GigaChat 429, retrying in %.0f seconds (attempt %d/%d)",
                    delay,
                    attempt,
                    self._MAX_RETRIES,
                )
                time.sleep(delay)
            try:
                return super()._generate(
                    messages, stop=stop, run_manager=run_manager, stream=stream, **kwargs
                )
            except ResponseError as e:
                if len(e.args) > 1 and e.args[1] == 429 and attempt < self._MAX_RETRIES:
                    last_exc = e
                    continue
                raise
        raise last_exc

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs,
    ) -> ChatResult:
        last_exc = None
        for attempt in range(self._MAX_RETRIES + 1):
            if attempt > 0:
                delay = self._RETRY_DELAYS[attempt - 1]
                logger.warning(
                    "GigaChat 429, retrying in %.0f seconds (attempt %d/%d)",
                    delay,
                    attempt,
                    self._MAX_RETRIES,
                )
                await asyncio.sleep(delay)
            try:
                return await super()._agenerate(
                    messages, stop=stop, run_manager=run_manager, stream=stream, **kwargs
                )
            except ResponseError as e:
                if len(e.args) > 1 and e.args[1] == 429 and attempt < self._MAX_RETRIES:
                    last_exc = e
                    continue
                raise
        raise last_exc


@configuration(GigaChatWithExtra)
class GigaChatModelFactory(ChatModelFactory[GigaChat], Configurable[GigaChatWithExtra]):

    def create_model(self) -> GigaChat:
        config = self.config

        config_dict = CredentialsUtils.inject_env_dict(
            keys=SENSITIVE_KEYS,
            conf=config.model_dump(exclude_none=True)
        )

        return GigaChatWithRetries(**config_dict)


@configuration(GigaChatEmbeddings)
class GigaChatEmbeddingsFactory(EmbeddingsModelFactory[GigaChatEmbeddings], Configurable[GigaChatEmbeddings]):
    def create_model(self) -> GigaChatEmbeddings:
        config = self.config
        config_dict = CredentialsUtils.inject_env_dict(
            keys=SENSITIVE_KEYS,
            conf=config.model_dump(exclude_none=True),
        )
        return config.model_copy(update=config_dict)
