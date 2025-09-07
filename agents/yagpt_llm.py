"""
Интеграция с YandexGPT API для LangChain
"""
import json
import requests
from typing import Any, Dict, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
import os
from dotenv import load_dotenv

load_dotenv()


class YandexGPT(LLM):
    """Кастомная LLM для работы с YandexGPT API"""
    
    api_key: str = Field(default_factory=lambda: os.getenv("YANDEX_API_KEY", ""))
    folder_id: str = Field(default_factory=lambda: os.getenv("YANDEX_FOLDER_ID", ""))
    model_id: str = Field(default_factory=lambda: os.getenv("YANDEX_MODEL_ID", "yandexgpt-lite"))
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2000)
    
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
        
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model_id}",
            "completionOptions": {
                "stream": False,
                "temperature": self.temperature,
                "maxTokens": str(self.max_tokens)
            },
            "messages": [
                {
                    "role": "user",
                    "text": prompt
                }
            ]
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result["result"]["alternatives"][0]["message"]["text"]
            
        except requests.exceptions.RequestException as e:
            return f"Ошибка API запроса: {str(e)}"
        except KeyError as e:
            return f"Ошибка парсинга ответа: {str(e)}"
        except Exception as e:
            return f"Неожиданная ошибка: {str(e)}"
