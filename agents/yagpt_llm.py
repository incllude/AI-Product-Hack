"""
Интеграция с YandexGPT API для LangGraph
Обновленная версия с улучшенной поддержкой LangGraph
"""
import json
import requests
from typing import Any, Dict, List, Optional, AsyncIterator
from langchain_core.language_models.llms import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult, Generation
from pydantic import Field
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
from datetime import datetime

load_dotenv()


class YandexGPT(LLM):
    """Кастомная LLM для работы с YandexGPT API (синхронная версия)"""
    
    api_key: str = Field(default_factory=lambda: os.getenv("YANDEX_API_KEY", ""))
    folder_id: str = Field(default_factory=lambda: os.getenv("YANDEX_FOLDER_ID", ""))
    model_id: str = Field(default_factory=lambda: os.getenv("YANDEX_MODEL_ID", "yandexgpt-lite"))
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2000)
    
    # Метаданные для LangGraph
    request_count: int = Field(default=0)
    total_tokens: int = Field(default=0)
    last_request_time: Optional[datetime] = Field(default=None)
    
    @property
    def _llm_type(self) -> str:
        return "yandex_gpt"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Вызов YandexGPT API"""
        
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }
        
        # Обновляем параметры из kwargs
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model_id}",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens)
            },
            "messages": [
                {
                    "role": "user",
                    "text": prompt
                }
            ]
        }
        
        try:
            # Обновляем метаданные
            self.request_count += 1
            self.last_request_time = datetime.now()
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            text_result = result["result"]["alternatives"][0]["message"]["text"]
            
            # Примерная оценка токенов (YandexGPT не возвращает точное количество)
            estimated_tokens = len(text_result.split()) * 1.3  # Приблизительная оценка
            self.total_tokens += int(estimated_tokens)
            
            return text_result
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Ошибка API запроса: {str(e)}"
            if run_manager:
                run_manager.on_llm_error(Exception(error_msg))
            return error_msg
        except KeyError as e:
            error_msg = f"Ошибка парсинга ответа: {str(e)}"
            if run_manager:
                run_manager.on_llm_error(Exception(error_msg))
            return error_msg
        except Exception as e:
            error_msg = f"Неожиданная ошибка: {str(e)}"
            if run_manager:
                run_manager.on_llm_error(Exception(error_msg))
            return error_msg
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования"""
        return {
            'request_count': self.request_count,
            'total_tokens': self.total_tokens,
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None,
            'model_id': self.model_id,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
    
    def reset_stats(self):
        """Сбрасывает статистику"""
        self.request_count = 0
        self.total_tokens = 0
        self.last_request_time = None


class YandexGPTChat(BaseChatModel):
    """Чат-модель для YandexGPT с поддержкой сообщений"""
    
    api_key: str = Field(default_factory=lambda: os.getenv("YANDEX_API_KEY", ""))
    folder_id: str = Field(default_factory=lambda: os.getenv("YANDEX_FOLDER_ID", ""))
    model_id: str = Field(default_factory=lambda: os.getenv("YANDEX_MODEL_ID", "yandexgpt-lite"))
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2000)
    
    # Метаданные для LangGraph
    request_count: int = Field(default=0)
    total_tokens: int = Field(default=0)
    last_request_time: Optional[datetime] = Field(default=None)
    
    @property
    def _llm_type(self) -> str:
        return "yandex_gpt_chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Генерация ответа на основе списка сообщений"""
        
        # Конвертируем сообщения в формат YandexGPT
        yandex_messages = self._convert_messages_to_yandex_format(messages)
        
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }
        
        # Обновляем параметры из kwargs
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model_id}",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens)
            },
            "messages": yandex_messages
        }
        
        try:
            # Обновляем метаданные
            self.request_count += 1
            self.last_request_time = datetime.now()
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            text_result = result["result"]["alternatives"][0]["message"]["text"]
            
            # Примерная оценка токенов
            estimated_tokens = len(text_result.split()) * 1.3
            self.total_tokens += int(estimated_tokens)
            
            # Создаем ChatGeneration
            message = AIMessage(content=text_result)
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            error_msg = f"Ошибка в YandexGPT Chat: {str(e)}"
            if run_manager:
                run_manager.on_llm_error(Exception(error_msg))
            
            # Возвращаем ошибку как сообщение
            error_message = AIMessage(content=error_msg)
            error_generation = ChatGeneration(message=error_message)
            return ChatResult(generations=[error_generation])
    
    def _convert_messages_to_yandex_format(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Конвертирует сообщения LangChain в формат YandexGPT"""
        yandex_messages = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                # YandexGPT не поддерживает системные сообщения напрямую,
                # поэтому добавляем их как пользовательские с префиксом
                yandex_messages.append({
                    "role": "user",
                    "text": f"Системная инструкция: {message.content}"
                })
            elif isinstance(message, HumanMessage):
                yandex_messages.append({
                    "role": "user",
                    "text": message.content
                })
            elif isinstance(message, AIMessage):
                yandex_messages.append({
                    "role": "assistant",
                    "text": message.content
                })
            else:
                # Для других типов сообщений используем пользовательскую роль
                yandex_messages.append({
                    "role": "user",
                    "text": str(message.content)
                })
        
        return yandex_messages
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования"""
        return {
            'request_count': self.request_count,
            'total_tokens': self.total_tokens,
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None,
            'model_id': self.model_id,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
    
    def reset_stats(self):
        """Сбрасывает статистику"""
        self.request_count = 0
        self.total_tokens = 0
        self.last_request_time = None


class AsyncYandexGPT:
    """Асинхронная версия YandexGPT для LangGraph"""
    
    def __init__(
        self,
        api_key: str = None,
        folder_id: str = None,
        model_id: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        self.api_key = api_key or os.getenv("YANDEX_API_KEY", "")
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID", "")
        self.model_id = model_id or os.getenv("YANDEX_MODEL_ID", "yandexgpt-lite")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Метаданные
        self.request_count = 0
        self.total_tokens = 0
        self.last_request_time = None
    
    async def agenerate(self, prompt: str, **kwargs) -> str:
        """Асинхронная генерация текста"""
        
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }
        
        # Обновляем параметры из kwargs
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model_id}",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens)
            },
            "messages": [
                {
                    "role": "user",
                    "text": prompt
                }
            ]
        }
        
        try:
            # Обновляем метаданные
            self.request_count += 1
            self.last_request_time = datetime.now()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    headers=headers, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
            
            text_result = result["result"]["alternatives"][0]["message"]["text"]
            
            # Примерная оценка токенов
            estimated_tokens = len(text_result.split()) * 1.3
            self.total_tokens += int(estimated_tokens)
            
            return text_result
            
        except Exception as e:
            return f"Ошибка в AsyncYandexGPT: {str(e)}"
    
    async def achat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Асинхронная генерация в чат-режиме"""
        
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {self.api_key}"
        }
        
        # Обновляем параметры из kwargs
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model_id}",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens)
            },
            "messages": messages
        }
        
        try:
            # Обновляем метаданные
            self.request_count += 1
            self.last_request_time = datetime.now()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    headers=headers, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
            
            text_result = result["result"]["alternatives"][0]["message"]["text"]
            
            # Примерная оценка токенов
            estimated_tokens = len(text_result.split()) * 1.3
            self.total_tokens += int(estimated_tokens)
            
            return text_result
            
        except Exception as e:
            return f"Ошибка в AsyncYandexGPT chat: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования"""
        return {
            'request_count': self.request_count,
            'total_tokens': self.total_tokens,
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None,
            'model_id': self.model_id,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
    
    def reset_stats(self):
        """Сбрасывает статистику"""
        self.request_count = 0
        self.total_tokens = 0
        self.last_request_time = None


# Функции-помощники для LangGraph

def create_yandex_llm(
    temperature: float = 0.7,
    max_tokens: int = 2000,
    model_id: str = None
) -> YandexGPT:
    """Создает экземпляр YandexGPT LLM"""
    return YandexGPT(
        temperature=temperature,
        max_tokens=max_tokens,
        model_id=model_id or os.getenv("YANDEX_MODEL_ID", "yandexgpt-lite")
    )

def create_yandex_chat(
    temperature: float = 0.7,
    max_tokens: int = 2000,
    model_id: str = None
) -> YandexGPTChat:
    """Создает экземпляр YandexGPT Chat"""
    return YandexGPTChat(
        temperature=temperature,
        max_tokens=max_tokens,
        model_id=model_id or os.getenv("YANDEX_MODEL_ID", "yandexgpt-lite")
    )

def create_async_yandex_llm(
    temperature: float = 0.7,
    max_tokens: int = 2000,
    model_id: str = None
) -> AsyncYandexGPT:
    """Создает экземпляр асинхронного YandexGPT"""
    return AsyncYandexGPT(
        temperature=temperature,
        max_tokens=max_tokens,
        model_id=model_id or os.getenv("YANDEX_MODEL_ID", "yandexgpt-lite")
    )

def validate_yandex_config() -> Dict[str, Any]:
    """Валидирует конфигурацию YandexGPT"""
    config = {
        'api_key': bool(os.getenv("YANDEX_API_KEY")),
        'folder_id': bool(os.getenv("YANDEX_FOLDER_ID")),
        'model_id': os.getenv("YANDEX_MODEL_ID", "yandexgpt-lite")
    }
    
    config['is_valid'] = config['api_key'] and config['folder_id']
    
    if not config['is_valid']:
        missing = []
        if not config['api_key']:
            missing.append('YANDEX_API_KEY')
        if not config['folder_id']:
            missing.append('YANDEX_FOLDER_ID')
        config['missing_vars'] = missing
    
    return config
