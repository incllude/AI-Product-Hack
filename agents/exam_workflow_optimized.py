"""
Оптимизированный workflow экзамена на LangGraph с быстрой инициализацией
"""
from typing import Dict, List, Optional, Any
from langgraph.graph import StateGraph, END
import asyncio
from concurrent.futures import ThreadPoolExecutor

from base import (
    ExamState, create_initial_exam_state, validate_exam_state, 
    update_exam_progress, should_continue_exam, calculate_exam_statistics
)
from question_agent import create_question_agent_langgraph
from evaluation_agent import create_evaluation_agent_langgraph
from diagnostic_agent import create_diagnostic_agent_langgraph
from theme_agent import create_theme_agent_langgraph
from topic_manager import TopicManager
from yagpt_llm import create_yandex_llm
from datetime import datetime
import uuid


class SharedLLMManager:
    """Singleton для управления общим экземпляром LLM"""
    _instance = None
    _llm = None
    
    @classmethod
    def get_shared_llm(cls):
        """Получает общий экземпляр YandexGPT LLM"""
        if cls._llm is None:
            print("🚀 Создание общего экземпляра YandexGPT...")
            cls._llm = create_yandex_llm()
            print("✅ YandexGPT инициализирован")
        return cls._llm
    
    @classmethod
    def reset_shared_llm(cls):
        """Сбрасывает общий экземпляр (для тестирования)"""
        cls._llm = None


class OptimizedExamWorkflowLangGraph:
    """Оптимизированный workflow экзамена на LangGraph с быстрой инициализацией"""
    
    def __init__(self, topic_info: Dict[str, Any] = None, max_questions: int = 5, use_theme_structure: bool = False):
        """
        Оптимизированная инициализация workflow экзамена
        
        Args:
            topic_info: Информация о теме экзамена (из TopicManager)
            max_questions: Максимальное количество вопросов
            use_theme_structure: Использовать ли тематическую структуру по таксономии Блума
        """
        print("🚀 Начало оптимизированной инициализации...")
        start_time = datetime.now()
        
        # Если тема не указана, используем тему по умолчанию
        if not topic_info:
            topic_manager = TopicManager()
            topic_info = topic_manager._get_default_topic()
        
        self.topic_info = topic_info
        self.difficulty = topic_info['difficulty']
        self.max_questions = max_questions
        self.use_theme_structure = use_theme_structure
        
        # Создание контекста для агентов
        topic_manager = TopicManager()
        self.topic_context = topic_manager.get_topic_context_for_prompts(topic_info)
        
        # Получаем общий экземпляр LLM
        self.shared_llm = SharedLLMManager.get_shared_llm()
        
        # Инициализация агентов с ленивой загрузкой
        self._agents_initialized = False
        self._theme_agent = None
        self._question_agent = None
        self._evaluation_agent = None
        self._diagnostic_agent = None
        
        # Создание workflow графа (легковесная операция)
        self.graph = self._create_exam_workflow_graph()
        self.app = self.graph.compile()
        
        # История workflow
        self.workflow_history = []
        
        end_time = datetime.now()
        init_duration = (end_time - start_time).total_seconds()
        print(f"✅ Оптимизированная инициализация завершена за {init_duration:.2f} секунд")
    
    def _initialize_agents_lazy(self):
        """Ленивая инициализация агентов при первом обращении"""
        if self._agents_initialized:
            return
        
        print("🔧 Ленивая инициализация агентов...")
        start_time = datetime.now()
        
        # Инициализация агентов с общим LLM
        if self.use_theme_structure:
            self._theme_agent = create_theme_agent_langgraph(
                topic_context=self.topic_context
            )
            # Заменяем LLM на общий экземпляр
            self._theme_agent.llm = self.shared_llm
        
        self._question_agent = create_question_agent_langgraph(
            difficulty=self.difficulty,
            topic_context=self.topic_context,
            theme_structure=None
        )
        # Заменяем LLM на общий экземпляр
        self._question_agent.llm = self.shared_llm
        
        self._evaluation_agent = create_evaluation_agent_langgraph(
            topic_context=self.topic_context
        )
        # Заменяем LLM на общий экземпляр
        self._evaluation_agent.llm = self.shared_llm
        
        self._diagnostic_agent = create_diagnostic_agent_langgraph(
            topic_context=self.topic_context
        )
        # Заменяем LLM на общий экземпляр
        self._diagnostic_agent.llm = self.shared_llm
        
        self._agents_initialized = True
        
        end_time = datetime.now()
        agents_duration = (end_time - start_time).total_seconds()
        print(f"✅ Агенты инициализированы за {agents_duration:.2f} секунд")
    
    async def _initialize_agents_async(self):
        """Асинхронная инициализация агентов для максимальной скорости"""
        if self._agents_initialized:
            return
        
        print("⚡ Асинхронная инициализация агентов...")
        start_time = datetime.now()
        
        # Функции для асинхронного создания агентов
        def create_theme_agent():
            if self.use_theme_structure:
                agent = create_theme_agent_langgraph(
                    topic_context=self.topic_context
                )
                # Заменяем LLM на общий экземпляр
                agent.llm = self.shared_llm
                return agent
            return None
        
        def create_question_agent():
            agent = create_question_agent_langgraph(
                difficulty=self.difficulty,
                topic_context=self.topic_context,
                theme_structure=None
            )
            # Заменяем LLM на общий экземпляр
            agent.llm = self.shared_llm
            return agent
        
        def create_evaluation_agent():
            agent = create_evaluation_agent_langgraph(
                topic_context=self.topic_context
            )
            # Заменяем LLM на общий экземпляр
            agent.llm = self.shared_llm
            return agent
        
        def create_diagnostic_agent():
            agent = create_diagnostic_agent_langgraph(
                topic_context=self.topic_context
            )
            # Заменяем LLM на общий экземпляр
            agent.llm = self.shared_llm
            return agent
        
        # Выполняем инициализацию параллельно
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_theme = executor.submit(create_theme_agent)
            future_question = executor.submit(create_question_agent)
            future_evaluation = executor.submit(create_evaluation_agent)
            future_diagnostic = executor.submit(create_diagnostic_agent)
            
            # Ждем завершения всех задач
            self._theme_agent = future_theme.result()
            self._question_agent = future_question.result()
            self._evaluation_agent = future_evaluation.result()
            self._diagnostic_agent = future_diagnostic.result()
        
        self._agents_initialized = True
        
        end_time = datetime.now()
        agents_duration = (end_time - start_time).total_seconds()
        print(f"✅ Агенты инициализированы асинхронно за {agents_duration:.2f} секунд")
    
    # Свойства с ленивой загрузкой
    @property
    def theme_agent(self):
        if not self._agents_initialized:
            self._initialize_agents_lazy()
        return self._theme_agent
    
    @property  
    def question_agent(self):
        if not self._agents_initialized:
            self._initialize_agents_lazy()
        return self._question_agent
    
    @property
    def evaluation_agent(self):
        if not self._agents_initialized:
            self._initialize_agents_lazy()
        return self._evaluation_agent
    
    @property
    def diagnostic_agent(self):
        if not self._agents_initialized:
            self._initialize_agents_lazy()
        return self._diagnostic_agent
    
    def force_initialize_agents(self):
        """Принудительная инициализация всех агентов"""
        self._initialize_agents_lazy()
    
    async def async_initialize_agents(self):
        """Асинхронная инициализация всех агентов"""
        await self._initialize_agents_async()
    
    def _create_exam_workflow_graph(self) -> StateGraph:
        """Создает граф workflow экзамена (легковесная операция)"""
        graph = StateGraph(ExamState)
        
        # Добавляем узлы workflow
        graph.add_node("initialize_exam", self._initialize_exam_node)
        graph.add_node("create_theme_structure", self._create_theme_structure_node)
        graph.add_node("start_exam", self._start_exam_node)
        graph.add_node("generate_question", self._generate_question_node)
        graph.add_node("wait_for_answer", self._wait_for_answer_node)
        graph.add_node("evaluate_answer", self._evaluate_answer_node)
        graph.add_node("update_progress", self._update_progress_node)
        graph.add_node("check_continuation", self._check_continuation_node)
        graph.add_node("complete_exam", self._complete_exam_node)
        graph.add_node("run_diagnostics", self._run_diagnostics_node)
        graph.add_node("finalize_results", self._finalize_results_node)
        
        # Определяем точку входа
        graph.set_entry_point("initialize_exam")
        
        # Определяем переходы
        graph.add_conditional_edges(
            "initialize_exam",
            self._should_create_theme_structure,
            {
                "create_theme": "create_theme_structure",
                "start_exam": "start_exam"
            }
        )
        
        graph.add_edge("create_theme_structure", "start_exam")
        graph.add_edge("start_exam", "generate_question")
        graph.add_edge("generate_question", "wait_for_answer")
        graph.add_edge("wait_for_answer", "evaluate_answer")
        graph.add_edge("evaluate_answer", "update_progress")
        
        graph.add_conditional_edges(
            "check_continuation",
            self._should_continue,
            {
                "continue": "generate_question",
                "complete": "complete_exam"
            }
        )
        
        graph.add_edge("update_progress", "check_continuation")
        graph.add_edge("complete_exam", "run_diagnostics")
        graph.add_edge("run_diagnostics", "finalize_results")
        graph.add_edge("finalize_results", END)
        
        return graph
    
    # Остальные методы остаются такими же, как в оригинальном workflow
    def _initialize_exam_node(self, state: ExamState) -> ExamState:
        """Узел инициализации экзамена"""
        state["status"] = "initialized"
        state["start_time"] = datetime.now()
        state["messages"].append("Экзамен инициализирован")
        return state
    
    def _should_create_theme_structure(self, state: ExamState) -> str:
        """Определяет, нужно ли создавать тематическую структуру"""
        return "create_theme" if state["use_theme_structure"] else "start_exam"
    
    def _create_theme_structure_node(self, state: ExamState) -> ExamState:
        """Узел создания тематической структуры"""
        if self.theme_agent:
            try:
                theme_structure = self.theme_agent.generate_theme_structure(
                    total_questions=state["max_questions"],
                    difficulty=state["difficulty"]
                )
                if not theme_structure.get("error"):
                    state["theme_structure"] = theme_structure
                    state["messages"].append("Тематическая структура создана")
                else:
                    state["errors"].append(f"Ошибка создания темы: {theme_structure['error']}")
            except Exception as e:
                state["errors"].append(f"Ошибка создания тематической структуры: {str(e)}")
        
        return state
    
    def _start_exam_node(self, state: ExamState) -> ExamState:
        """Узел запуска экзамена"""
        state["status"] = "in_progress"
        state["messages"].append("Экзамен начат")
        return state
    
    def _generate_question_node(self, state: ExamState) -> ExamState:
        """Узел генерации вопроса"""
        # Агент инициализируется автоматически при обращении к свойству
        current_question_number = len(state["questions"]) + 1
        
        try:
            question_data = self.question_agent.generate_question(
                current_question_number,
                state["evaluation_summaries"]
            )
            
            if not question_data.get("error"):
                question_data["question_number"] = current_question_number
                question_data["timestamp"] = datetime.now()
                state["questions"].append(question_data)
                state["current_question_number"] = current_question_number
                state["messages"].append(f"Вопрос {current_question_number} сгенерирован")
            else:
                state["errors"].append(f"Ошибка генерации вопроса: {question_data['error']}")
                
        except Exception as e:
            state["errors"].append(f"Ошибка генерации вопроса: {str(e)}")
        
        return state
    
    def _wait_for_answer_node(self, state: ExamState) -> ExamState:
        """Узел ожидания ответа (заглушка для интерактивного режима)"""
        state["messages"].append("Ожидание ответа студента")
        return state
    
    def _evaluate_answer_node(self, state: ExamState) -> ExamState:
        """Узел оценки ответа"""
        # Этот метод будет вызван после получения ответа от студента
        return state
    
    def _update_progress_node(self, state: ExamState) -> ExamState:
        """Узел обновления прогресса"""
        state = update_exam_progress(state)
        return state
    
    def _check_continuation_node(self, state: ExamState) -> ExamState:
        """Узел проверки продолжения экзамена"""
        return state
    
    def _should_continue(self, state: ExamState) -> str:
        """Определяет, следует ли продолжить экзамен"""
        return "continue" if should_continue_exam(state) else "complete"
    
    def _complete_exam_node(self, state: ExamState) -> ExamState:
        """Узел завершения экзамена"""
        state["status"] = "completed"
        state["end_time"] = datetime.now()
        state["messages"].append("Экзамен завершен")
        return state
    
    def _run_diagnostics_node(self, state: ExamState) -> ExamState:
        """Узел запуска диагностики"""
        try:
            diagnostic_result = self.diagnostic_agent.run_diagnostics(
                state["questions"],
                state["evaluations"]
            )
            
            if not diagnostic_result.get("error"):
                state["diagnostic_result"] = diagnostic_result
                state["messages"].append("Диагностика выполнена")
            else:
                state["errors"].append(f"Ошибка диагностики: {diagnostic_result['error']}")
                
        except Exception as e:
            state["errors"].append(f"Ошибка запуска диагностики: {str(e)}")
        
        return state
    
    def _finalize_results_node(self, state: ExamState) -> ExamState:
        """Узел финализации результатов"""
        state["metadata"]["final_statistics"] = calculate_exam_statistics(state)
        state["messages"].append("Результаты финализированы")
        return state
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Получение статистики производительности"""
        return {
            "agents_initialized": self._agents_initialized,
            "shared_llm_stats": {
                "request_count": self.shared_llm.request_count if self.shared_llm else 0,
                "total_tokens": self.shared_llm.total_tokens if self.shared_llm else 0,
                "last_request": self.shared_llm.last_request_time if self.shared_llm else None
            },
            "workflow_history_length": len(self.workflow_history)
        }


# Функция для создания оптимизированного workflow
def create_optimized_exam_workflow(
    topic_info: Dict[str, Any] = None,
    max_questions: int = 5,
    use_theme_structure: bool = False
) -> OptimizedExamWorkflowLangGraph:
    """Создает экземпляр оптимизированного ExamWorkflow на LangGraph"""
    return OptimizedExamWorkflowLangGraph(
        topic_info=topic_info,
        max_questions=max_questions,
        use_theme_structure=use_theme_structure
    )
