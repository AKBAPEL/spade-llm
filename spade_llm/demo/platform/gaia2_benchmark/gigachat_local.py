import os
import time
import json
import re
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any, Union
from langchain_gigachat import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
import uvicorn
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="GigaChat ARE Adapter", version="6.0")

GIGACHAT_TOKEN = os.environ['GIGA_CRED']
print(f"Token loaded: {GIGACHAT_TOKEN[:5]}...")

# Инициализация GigaChat
llm = GigaChat(
    credentials=GIGACHAT_TOKEN,
    scope="GIGACHAT_API_CORP",
    model="GigaChat-2-Max",
    verify_ssl_certs=False,
    temperature=0.0,
    max_tokens=8192
)


# === Models ===

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    model: str = "gigachat"
    messages: List[ChatMessage]
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict]] = "auto"
    functions: Optional[List[FunctionDefinition]] = None
    function_call: Optional[Union[str, Dict]] = None
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


# === Helper functions ===

def convert_messages_to_langchain(messages: List[ChatMessage]):
    """Конвертирует сообщения в формат LangChain"""
    result = []
    for msg in messages:
        if msg.role == "system":
            result.append(SystemMessage(content=msg.content or ""))
        elif msg.role == "user":
            result.append(HumanMessage(content=msg.content or ""))
        elif msg.role == "assistant":
            result.append(AIMessage(content=msg.content or ""))
        elif msg.role == "tool":
            result.append(ToolMessage(
                content=msg.content or "",
                tool_call_id=msg.tool_call_id or "",
                name=msg.name or ""
            ))
    return result


def create_gigachat_function_declarations(tools_info: List[Dict]) -> List[Dict]:
    """Создает объявления функций для GigaChat API"""
    declarations = []
    for info in tools_info:
        properties = {}
        required = []
        for param_name, param_info in info["parameters"].items():
            prop_type = param_info.get("type", "string")
            json_type = "string"
            if prop_type == "integer":
                json_type = "integer"
            elif prop_type == "number":
                json_type = "number"
            elif prop_type == "boolean":
                json_type = "boolean"
            properties[param_name] = {
                "type": json_type,
                "description": f"Parameter {param_name}"
            }
            required.append(param_name)

        parameters = {
            "type": "object",
            "properties": properties
        }
        if required:
            parameters["required"] = required

        declarations.append({
            "name": info["name"],
            "description": info["description"][:500],
            "parameters": parameters
        })
    return declarations


# === Endpoints ===

@app.get("/")
async def root():
    return {
        "status": "GigaChat ARE Adapter",
        "model": "gigachat"
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "gigachat",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "sber"
        }]
    }


@app.post("/v1/chat/completions")
async def chat_completions(
        request: ChatCompletionRequest,
        authorization: Optional[str] = Header(None)
):
    print(f"\n{'=' * 60}")
    print(f"[REQUEST] Messages: {len(request.messages)}")
    print(f"[REQUEST] tools present: {bool(request.tools)}")
    print(f"[REQUEST] functions present: {bool(request.functions)}")

    try:
        messages = convert_messages_to_langchain(request.messages)

        # Режим function calling только если явно запрошены инструменты
        if request.tools or request.functions:
            print(f"[MODE] Explicit function calling")

            # Собираем информацию о функциях
            tools_info = []
            if request.tools:
                for tool in request.tools:
                    tools_info.append({
                        "name": tool.function.name,
                        "description": tool.function.description or "",
                        "parameters": tool.function.parameters.get("properties", {})
                    })
            elif request.functions:
                for func in request.functions:
                    tools_info.append({
                        "name": func.name,
                        "description": func.description or "",
                        "parameters": func.parameters.get("properties", {})
                    })

            if tools_info:
                function_declarations = create_gigachat_function_declarations(tools_info)
                llm_with_functions = llm.bind(
                    functions=function_declarations,
                    function_call="auto"
                )
                response = llm_with_functions.invoke(messages)

                function_call = None
                if hasattr(response, 'additional_kwargs') and response.additional_kwargs:
                    function_call = response.additional_kwargs.get('function_call')

                if function_call:
                    tool_name = function_call.get('name', '')
                    tool_args = function_call.get('arguments', {})
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except:
                            tool_args = {}

                    # Формируем ответ с tool_calls (OpenAI-совместимый)
                    tool_calls = [{
                        "id": f"call_{int(time.time())}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args, ensure_ascii=False)
                        }
                    }]

                    return ChatCompletionResponse(
                        id=f"chatcmpl-{int(time.time())}",
                        created=int(time.time()),
                        model=request.model,
                        choices=[ChatCompletionChoice(
                            message=ChatMessage(
                                role="assistant",
                                content=response.content,  # может быть None или текстом
                                tool_calls=tool_calls
                            ),
                            finish_reason="tool_calls"
                        )],
                        usage=ChatCompletionUsage(
                            prompt_tokens=len(str(messages).split()),
                            completion_tokens=len(str(tool_args).split())
                        )
                    )
                else:
                    # Функция не вызвана, возвращаем текст
                    return ChatCompletionResponse(
                        id=f"chatcmpl-{int(time.time())}",
                        created=int(time.time()),
                        model=request.model,
                        choices=[ChatCompletionChoice(
                            message=ChatMessage(role="assistant", content=response.content),
                            finish_reason="stop"
                        )],
                        usage=ChatCompletionUsage(
                            prompt_tokens=len(str(messages).split()),
                            completion_tokens=len(response.content.split())
                        )
                    )
            else:
                # Нет инструментов (странно), просто вызываем LLM
                print("\n[CONTEXT] Messages sent to model:")
                for i, msg in enumerate(messages):
                    print(f"  {i}: {msg.__class__.__name__} - {msg.content}" if len(
                        msg.content) > 200 else f"  {i}: {msg.__class__.__name__} - {msg.content}")
                response = llm.invoke(messages)
                print(f"\n[MODEL RAW RESPONSE] {response.content}")
                # Если нужно больше деталей (например, есть ли function_call в дополнительных kwargs):
                print(f"[MODEL ADDITIONAL_KWARGS] {response.additional_kwargs}")
                return ChatCompletionResponse(
                    id=f"chatcmpl-{int(time.time())}",
                    created=int(time.time()),
                    model=request.model,
                    choices=[ChatCompletionChoice(
                        message=ChatMessage(role="assistant", content=response.content),
                        finish_reason="stop"
                    )],
                    usage=ChatCompletionUsage(
                        prompt_tokens=len(str(messages).split()),
                        completion_tokens=len(response.content.split())
                    )
                )
        else:
            # Обычный запрос (в том числе ARE) — просто вызываем LLM без функций
            print(f"[MODE] Simple completion (ARE or generic)")
            print("\n[CONTEXT] Messages sent to model:")
            for i, msg in enumerate(messages):
                print(f"  {i}: {msg.__class__.__name__} - {msg.content}" if len(
                    msg.content) > 200 else f"  {i}: {msg.__class__.__name__} - {msg.content}")
            response = llm.invoke(messages)
            print(f"\n[MODEL RAW RESPONSE] {response.content}")
            # Если нужно больше деталей (например, есть ли function_call в дополнительных kwargs):
            print(f"[MODEL ADDITIONAL_KWARGS] {response.additional_kwargs}")

            return ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=request.model,
                choices=[ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=response.content),
                    finish_reason="stop"
                )],
                usage=ChatCompletionUsage(
                    prompt_tokens=len(str(messages).split()),
                    completion_tokens=len(response.content.split())
                )
            )

    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)