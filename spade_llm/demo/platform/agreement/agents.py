import logging
from asyncio import sleep as asleep
import asyncio
from spade_llm.core.api import AgentContext
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from pydantic.fields import Field
from spade_llm.core.agent import Agent
from spade_llm.core.behaviors import MessageHandlingBehavior, MessageTemplate, ContextBehaviour
from spade_llm.core.conf import configuration, Configurable
from spade_llm import consts
import requests
import uvicorn
from fastapi import FastAPI
import json
import uuid
from typing import Dict, List
from fastapi import Request
from sse_starlette.sse import EventSourceResponse
import re

logger = logging.getLogger(__name__)


class PlatformAgentConf(BaseModel):
    model: str = Field(description="Model name")
    senat_agents: List[str] = Field(description="List of agent names")


class ApiRequestBehaviour(ContextBehaviour):
    def __init__(self, context: AgentContext, config: PlatformAgentConf, request: str, ids: List[str],
                 request_obj: Request):
        super().__init__(context)
        self.config = config
        self.request = request
        self.ids = ids
        self.agent_map = {'2': 'second', '3': 'third', '4': 'fourth', '5': 'fifth', '6': 'sixth', '7': 'seventh'}
        self.request_obj = request_obj
        self.event_queue = asyncio.Queue()
        print(f"Запрос: {request}")

    async def send_event(self, data: Dict):
        """Отправка события в очередь, разбивая предложения на подгруппы слов для type='reasoning'"""
        event_id = str(uuid.uuid4())  # Генерируем один ID для всех частей текста
        text = data.get("text", "")
        tag = data.get("tag", "0")
        event_type = data.get("type", "text")
        status = data.get("status", "running")

        if event_type == "reasoning":
            # Разбиваем текст на предложения и переносы строк
            parts = re.split(r'((?:[^.!?]+[.!?]+(?:\s+|$))|(\n+))', text)
            parts = [part for part in parts if part]  # Удаляем пустые элементы

            if not parts:
                # Если текст пустой, отправляем одно событие
                event = {
                    "event": "message",
                    "id": event_id,
                    "retry": 15000,
                    "data": json.dumps({
                        "status": status,
                        "text": "",
                        "type": event_type,
                        "tag": tag
                    }, ensure_ascii=False)
                }
                await self.event_queue.put(event)
                return

            group_size = 2  # Количество слов в подгруппе
            for part in parts:
                if re.match(r'\n+', part):  # Отправляем перенос строки как отдельное событие
                    event = {
                        "event": "message",
                        "id": event_id,
                        "retry": 15000,
                        "data": json.dumps({
                            "status": status,
                            "text": part,
                            "type": event_type,
                            "tag": tag
                        }, ensure_ascii=False)
                    }
                    await self.event_queue.put(event)
                    await asyncio.sleep(0.1)
                    continue

                # Разбиваем предложение на слова, пробелы и знаки препинания
                sub_parts = re.findall(r'(\w+|[^\w\s]|\s+)', part)
                sub_parts = [sp for sp in sub_parts if sp]
                current_group = []
                word_count = 0

                for sp in sub_parts:
                    current_group.append(sp)
                    if re.match(r'\w+', sp):
                        word_count += 1
                    if word_count >= group_size or re.match(r'[.!?]', sp):  # Формируем группу или при конце предложения
                        event = {
                            "event": "message",
                            "id": event_id,
                            "retry": 15000,
                            "data": json.dumps({
                                "status": status,
                                "text": ''.join(current_group),
                                "type": event_type,
                                "tag": tag
                            }, ensure_ascii=False)
                        }
                        await self.event_queue.put(event)
                        await asyncio.sleep(0.1)
                        current_group = []
                        word_count = 0
                if current_group:  # Отправляем остаток
                    event = {
                        "event": "message",
                        "id": event_id,
                        "retry": 15000,
                        "data": json.dumps({
                            "status": status,
                            "text": ''.join(current_group),
                            "type": event_type,
                            "tag": tag
                        }, ensure_ascii=False)
                    }
                    await self.event_queue.put(event)
                    await asyncio.sleep(0.1)
        else:
            # Для всех остальных типов отправляем текст целиком
            event = {
                "event": "message",
                "id": event_id,
                "retry": 15000,
                "data": json.dumps(data, ensure_ascii=False)
            }
            await self.event_queue.put(event)

    async def event_generator(self):
        """Генератор событий для SSE"""
        while not self.is_done():
            if await self.request_obj.is_disconnected():
                print("Client disconnected")
                break
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=0.35)
                yield event
            except asyncio.TimeoutError:
                continue

    async def step(self) -> None:
        print("in STEP")
        answers = []

        for s_agent in self.ids:
            if s_agent in self.agent_map.keys():
                thread = await self.context.fork_thread()
                await (thread.inform(self.agent_map[s_agent]).with_content(self.request))
                opinion_receiver = await self.receive(
                    MessageTemplate(thread.thread_id),
                    timeout=60
                )
                await self.send_event(
                    {"status": "running", "text": opinion_receiver.content, "type": "reasoning", "tag": s_agent})
        # Тут разграничение между обсуждением и принятием решения

        for s_agent in self.ids:
            if s_agent in self.agent_map.keys():
                thread = await self.context.fork_thread()
                await (thread.request(self.agent_map[s_agent]).with_content(self.request))

                receiver = await self.receive(
                    MessageTemplate(thread.thread_id),
                    timeout=60
                )
                response = VotingResponse.model_validate_json(receiver.content)
                answers.append(response.vote)
                await self.send_event(
                    {"status": "running", "text": response.explanation, "type": "user", "tag": s_agent})
                await self.send_event(
                    {"status": "running", "text": response.vote, "type": "text", "tag": s_agent})

        if sum([1 if 'за' in el.lower() else 0 for el in answers]) / len(answers) > 0.6:
            print("Акт подписан")
            await self.send_event(
                {"status": "running", "text": f"Акт подписан", "type": "text", "tag": "0"})
        else:
            print("Акт не подписан")
            await self.send_event(
                {"status": "running", "text": f"Акт не подписан", "type": "text", "tag": "0"})
        await self.send_event(
            {"status": "done", "text": "", "type": "text", "tag": "0"})
        await self.context.propose("communication_board").with_content('')
        blank_receiver = await self.receive(
            MessageTemplate(thread_id=self.context.thread_id, performative=consts.INFORM),
            timeout=60
        )
        await asleep(1)
        self.set_is_done()


class FrontVotingRequest(BaseModel):
    auth_key: str = Field(description="Authorization key")
    person_list: List[str] = Field(description="List of person names")
    document_name: str = Field(description="Document name")


@configuration(PlatformAgentConf)
class PlatformAgent(Agent, Configurable[PlatformAgentConf]):
    class ApiBehaviour(ContextBehaviour):
        def __init__(self, context: AgentContext, config: PlatformAgentConf):
            super().__init__(context)
            self.config = config

        async def step(self) -> None:
            app = FastAPI()

            @app.post("/senat")
            async def senat_stream(data: FrontVotingRequest, request: Request):
                thread = await self.context.fork_thread()
                api_request = ApiRequestBehaviour(context=thread, config=self.config, request=data.document_name,
                                                  ids=data.person_list, request_obj=request)
                self.agent.add_behaviour(api_request)
                # await api_request.join()
                return EventSourceResponse(
                    api_request.event_generator(),
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    }
                )

            config = uvicorn.Config(app, host="0.0.0.0", port=8001, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()

    def setup(self):
        self.add_behaviour(self.ApiBehaviour(self.default_context, self.config))


class VotingResponse(BaseModel):
    vote: str = Field(
        description="Ответ в виде строки. Отвечай 'за' если согласен, 'против' если не согласен, 'воздержусь' если не знаешь.")
    explanation: str = Field(description="Explaination of the answer", default='')


class FirstAgentConf(BaseModel):
    pass


class CommunicationBoardAgentConf(BaseModel):
    dialogue_history: Dict[str, str] = Field(
        description="Senat dialogue history in key-value format",
        default={}
    )


@configuration(CommunicationBoardAgentConf)
class CommunicationBoardAgent(Agent, Configurable[CommunicationBoardAgentConf]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    class ReturnDialogBehaviour(MessageHandlingBehavior):
        def __init__(self, config: CommunicationBoardAgentConf):
            super().__init__(MessageTemplate.request())
            self.config = config

        async def step(self) -> None:
            # Return the current dialogue history as a JSON string
            history_model = SenatOpinionHistory(dialogue_history=self.config.dialogue_history)
            await self.context.reply_with_acknowledge(self.message).with_content(history_model.model_dump_json())

    class ExpandDialogBehaviour(MessageHandlingBehavior):
        def __init__(self, config: CommunicationBoardAgentConf):
            super().__init__(MessageTemplate.inform())
            self.config = config

        async def step(self) -> None:
            # Add new message to the dialogue history with a timestamp as key
            parser = PydanticOutputParser(pydantic_object=PersonOpinion)
            person = parser.parse(self.message.content)
            self.config.dialogue_history[person.id] = person.opinion

    class ClearDialogBehaviour(MessageHandlingBehavior):
        def __init__(self, config: CommunicationBoardAgentConf):
            super().__init__(MessageTemplate.propose())
            self.config = config

        async def step(self) -> None:
            self.config.dialogue_history.clear()
            await self.context.reply_with_inform(self.message).with_content('')

    def setup(self):
        self.add_behaviour(self.ReturnDialogBehaviour(self.config))
        self.add_behaviour(self.ExpandDialogBehaviour(self.config))
        self.add_behaviour(self.ClearDialogBehaviour(self.config))


class QueryRequest(BaseModel):
    auth_key: str
    work_mode: str
    person_id: int
    query_text: str
    messages: Dict[str, str]


class SenatOpinionHistory(BaseModel):
    dialogue_history: Dict[str, str] = Field(
        description="Senat dialogue history in key-value format",
        default={}
    )


class PersonOpinion(BaseModel):
    id: str = Field(description="Person id")
    opinion: str = Field(description="Person opinion")


class LLMOutputFormat(BaseModel):
    answer: str = Field(description="Answer to the query")
    explanation: str = Field(description="Explanation of the answer")


class AgentOpinionBehaviour(MessageHandlingBehavior):
    def __init__(self, agent_num: str):
        super().__init__(MessageTemplate.inform())
        self.agent_num = agent_num

    async def step(self):
        request = self.message.content
        await self.context.request("communication_board").with_content('')
        receiver = await self.receive(
            MessageTemplate(thread_id=self.context.thread_id, performative=consts.ACKNOWLEDGE),
            timeout=60
        )
        gigachat_got_answer = False
        while not gigachat_got_answer:
            try:
                # Parse the received dialogue history
                dialogue_history = SenatOpinionHistory.model_validate_json(receiver.content)
                print('Received dialogue history:', dialogue_history)

                data = QueryRequest(
                    auth_key="LWVCWoR85ZEGmPF6WW2D",
                    work_mode="senate_opinion",
                    person_id=int(self.agent_num),
                    query_text=request,
                    messages=dialogue_history.dialogue_history
                )

                response = requests.post(
                    "http://localhost:7004/analyze_query",
                    json=data.model_dump(),
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    print("✅ Request successful!")
                    result = response.json()
                    print(f"Agent id: {self.agent_num}", "Server response:", result)
                    opinion = PersonOpinion(id=self.agent_num, opinion=result['opinion'])
                    await self.context.inform("communication_board").with_content(opinion)
                    await self.context.reply_with_acknowledge(self.message).with_content(result['opinion'])
                    gigachat_got_answer = True
                else:
                    print(f"❌ Error! Code: {response.status_code}")
                    print("Server response:", response.text)
                    await asleep(1)
            except Exception as e:
                print("⚠️ Error processing request:", e)
                await asleep(1)


class HandleAgentRequestBehaviour(MessageHandlingBehavior):
    def __init__(self, agent_num: str):
        super().__init__(MessageTemplate.request())
        self.agent_num = agent_num

    async def step(self):
        request = self.message.content
        await self.context.request("communication_board").with_content('')
        receiver = await self.receive(
            MessageTemplate(thread_id=self.context.thread_id, performative=consts.ACKNOWLEDGE),
            timeout=60
        )
        gigachat_got_answer = False
        while not gigachat_got_answer:
            try:
                # Parse the received dialogue history
                dialogue_history = SenatOpinionHistory.model_validate_json(receiver.content)

                data = QueryRequest(
                    auth_key="LWVCWoR85ZEGmPF6WW2D",
                    work_mode="senate_voting",
                    person_id=int(self.agent_num),
                    query_text=request,
                    messages=dialogue_history.dialogue_history
                )

                response = requests.post(
                    "http://localhost:7004/analyze_query",
                    json=data.model_dump(),
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    print("✅ Request successful!")
                    result = response.json()
                    print(f"Agent id: {self.agent_num}", "Server response:", result)
                    #  Тут Жесть с форматом ответа
                    vote = " " + result['answer'].capitalize()
                    await self.context.reply_with_acknowledge(self.message).with_content(
                        VotingResponse(vote=vote, explanation=result['explanation']))
                    gigachat_got_answer = True
                else:
                    print(f"❌ Error! Code: {response.status_code}")
                    print("Server response:", response.text)
                    await asleep(1)
            except Exception as e:
                print("⚠️ Error processing request:", e)
                await asleep(1)


@configuration(FirstAgentConf)
class FirstAgent(Agent, Configurable[FirstAgentConf]):
    def setup(self):
        self.add_behaviour(HandleAgentRequestBehaviour(
            agent_num='1'
        ))
        self.add_behaviour(AgentOpinionBehaviour(
            agent_num='1'
        ))


class SecondAgentConf(BaseModel):
    pass


@configuration(SecondAgentConf)
class SecondAgent(Agent, Configurable[SecondAgentConf]):
    def setup(self):
        self.add_behaviour(HandleAgentRequestBehaviour(
            agent_num='2'
        ))
        self.add_behaviour(AgentOpinionBehaviour(
            agent_num='2'
        ))


class ThirdAgentConf(BaseModel):
    pass


@configuration(ThirdAgentConf)
class ThirdAgent(Agent, Configurable[ThirdAgentConf]):
    def setup(self):
        self.add_behaviour(HandleAgentRequestBehaviour(
            agent_num='3'
        ))
        self.add_behaviour(AgentOpinionBehaviour(
            agent_num='3'
        ))


class FourthAgentConf(BaseModel):
    pass


@configuration(FourthAgentConf)
class FourthAgent(Agent, Configurable[FourthAgentConf]):
    def setup(self):
        self.add_behaviour(HandleAgentRequestBehaviour(
            agent_num='4'
        ))
        self.add_behaviour(AgentOpinionBehaviour(
            agent_num='4'
        ))


class FifthAgentConf(BaseModel):
    pass


@configuration(FifthAgentConf)
class FifthAgent(Agent, Configurable[FifthAgentConf]):
    def setup(self):
        self.add_behaviour(HandleAgentRequestBehaviour(
            agent_num='5'
        ))
        self.add_behaviour(AgentOpinionBehaviour(
            agent_num='5'
        ))


class SixthAgentConf(BaseModel):
    pass


@configuration(SixthAgentConf)
class SixthAgent(Agent, Configurable[SixthAgentConf]):
    def setup(self):
        self.add_behaviour(HandleAgentRequestBehaviour(
            agent_num='6'
        ))
        self.add_behaviour(AgentOpinionBehaviour(
            agent_num='6'
        ))


class SeventhAgentConf(BaseModel):
    pass


@configuration(SeventhAgentConf)
class SeventhAgent(Agent, Configurable[SeventhAgentConf]):
    def setup(self):
        self.add_behaviour(HandleAgentRequestBehaviour(
            agent_num='7'
        ))
        self.add_behaviour(AgentOpinionBehaviour(
            agent_num='7'
        ))
