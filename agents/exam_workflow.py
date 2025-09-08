
"""
Основной workflow экзамена на LangGraph, объединяющий все агенты
"""
from typing import Dict, List, Optional, Any
from langgraph.graph import StateGraph, END
# ToolExecutor больше не используется в новых версиях LangGraph

from base import (
    ExamState, create_initial_exam_state, validate_exam_state, 
    update_exam_progress, should_continue_exam, calculate_exam_statistics
)
from question_agent import create_question_agent_langgraph
from evaluation_agent import create_evaluation_agent_langgraph
from diagnostic_agent import create_diagnostic_agent_langgraph
from theme_agent import create_theme_agent_langgraph
from topic_manager import TopicManager
from datetime import datetime
import uuid


class ExamWorkflowLangGraph:
    """Основной workflow экзамена на LangGraph"""
    
    def __init__(self, topic_info: Dict[str, Any] = None, max_questions: int = 5, use_theme_structure: bool = False):
        """
        Инициализация workflow экзамена
        
        Args:
            topic_info: Информация о теме экзамена (из TopicManager)
            max_questions: Максимальное количество вопросов
            use_theme_structure: Использовать ли тематическую структуру по таксономии Блума
        """
        # Если тема не указана, используем тему по умолчанию
        if not topic_info:
            topic_manager = TopicManager()
            topic_info = topic_manager._get_default_topic()
        
        self.topic_info = topic_info
        self.subject = topic_info['subject']
        self.difficulty = topic_info['difficulty']
        self.max_questions = max_questions
        self.use_theme_structure = use_theme_structure
        
        # Создание контекста для агентов
        topic_manager = TopicManager()
        self.topic_context = topic_manager.get_topic_context_for_prompts(topic_info)
        
        # Инициализация агентов
        self._initialize_agents()
        
        # Создание workflow графа
        self.graph = self._create_exam_workflow_graph()
        self.app = self.graph.compile()
        
        # История workflow
        self.workflow_history = []
    
    def _initialize_agents(self):
        """Инициализирует всех агентов"""
        # Создание тематического агента (если нужно)
        self.theme_agent = None
        self.theme_structure = None
        
        if self.use_theme_structure:
            self.theme_agent = create_theme_agent_langgraph(
                subject=self.subject,
                topic_context=self.topic_context
            )
        
        # Создание основных агентов
        self.question_agent = create_question_agent_langgraph(
            subject=self.subject,
            difficulty=self.difficulty,
            topic_context=self.topic_context,
            theme_structure=self.theme_structure
        )
        
        self.evaluation_agent = create_evaluation_agent_langgraph(
            subject=self.subject,
            topic_context=self.topic_context
        )
        
        self.diagnostic_agent = create_diagnostic_agent_langgraph(
            subject=self.subject,
            topic_context=self.topic_context
        )
    
    def _create_exam_workflow_graph(self) -> StateGraph:
        """Создает граф workflow экзамена"""
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
        
        # Добавляем условные ребра
        graph.add_conditional_edges(
            "initialize_exam",
            self._decide_theme_creation,
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
            "update_progress",
            self._decide_next_action,
            {
                "continue": "generate_question",
                "complete": "complete_exam"
            }
        )
        
        graph.add_edge("complete_exam", "run_diagnostics")
        graph.add_edge("run_diagnostics", "finalize_results")
        
        # Завершение
        graph.add_edge("finalize_results", END)
        
        return graph
    
    def _initialize_exam_node(self, state: ExamState) -> ExamState:
        """Инициализирует экзамен"""
        try:
            # Валидируем состояние
            validation_errors = validate_exam_state(state)
            if validation_errors:
                state["errors"].extend(validation_errors)
                return state
            
            # Обновляем метаданные
            state["metadata"].update({
                "workflow_initialized": datetime.now(),
                "topic_info": self.topic_info,
                "agents_initialized": True
            })
            
            state["messages"].append("Workflow экзамена инициализирован")
            
            return state
            
        except Exception as e:
            error_msg = f"Ошибка инициализации экзамена: {str(e)}"
            state["errors"].append(error_msg)
            return state
    
    def _decide_theme_creation(self, state: ExamState) -> str:
        """Решает, нужно ли создавать тематическую структуру"""
        if state.get("errors"):
            return "start_exam"  # Пропускаем создание темы при ошибках
        
        if state.get("use_theme_structure", False) and not state.get("theme_structure"):
            return "create_theme"
        else:
            return "start_exam"
    
    def _create_theme_structure_node(self, state: ExamState) -> ExamState:
        """Создает тематическую структуру экзамена"""
        try:
            if not self.theme_agent:
                state["errors"].append("ThemeAgent не инициализирован")
                return state
            
            # Генерируем тематическую структуру
            theme_structure = self.theme_agent.generate_theme_structure(
                total_questions=state["max_questions"],
                difficulty=state["difficulty"]
            )
            
            if theme_structure.get("error"):
                state["errors"].append(f"Ошибка создания тематической структуры: {theme_structure['error']}")
                return state
            
            state["theme_structure"] = theme_structure
            
            # Обновляем агента вопросов с новой тематической структурой
            self.question_agent.theme_structure = theme_structure
            
            state["messages"].append("Тематическая структура создана")
            
            return state
            
        except Exception as e:
            error_msg = f"Ошибка создания тематической структуры: {str(e)}"
            state["errors"].append(error_msg)
            return state
    
    def _start_exam_node(self, state: ExamState) -> ExamState:
        """Начинает экзамен"""
        try:
            state["status"] = "in_progress"
            state["start_time"] = datetime.now()
            
            state["messages"].append(f"Экзамен начат для {state['student_name']}")
            
            return state
            
        except Exception as e:
            error_msg = f"Ошибка начала экзамена: {str(e)}"
            state["errors"].append(error_msg)
            return state
    
    def _generate_question_node(self, state: ExamState) -> ExamState:
        """Генерирует следующий вопрос"""
        try:
            current_question_number = len(state["questions"]) + 1
            
            # Получаем характеристики оценок БЕЗ текстов ответов для QuestionAgent
            evaluation_summaries = self.evaluation_agent.get_evaluation_summaries_for_question_agent()
            
            # Генерация вопроса на основе характеристик (НЕ текстов ответов)
            question_data = self.question_agent.generate_question(
                current_question_number, 
                evaluation_summaries  # Только характеристики, БЕЗ текстов ответов
            )
            
            if question_data.get("error"):
                state["errors"].append(f"Ошибка генерации вопроса: {question_data['error']}")
                return state
            
            # Добавление метаданных о приватности
            question_data.update({
                'question_number': current_question_number,
                'timestamp': datetime.now(),
                'privacy_protected': True,  # Подтверждение соблюдения приватности
                'evaluation_summaries_count': len(evaluation_summaries),
                'data_flow': 'EvaluationAgent → characteristics → QuestionAgent'
            })
            
            state["questions"].append(question_data)
            state["current_question_number"] = current_question_number
            
            state["messages"].append(f"Сгенерирован вопрос {current_question_number}")
            
            return state
            
        except Exception as e:
            error_msg = f"Ошибка генерации вопроса: {str(e)}"
            state["errors"].append(error_msg)
            return state
    
    def _wait_for_answer_node(self, state: ExamState) -> ExamState:
        """Ожидает ответ от студента (placeholder для интерактивности)"""
        try:
            # В реальной реализации здесь будет ожидание ввода пользователя
            # Для демонстрации workflow просто добавляем сообщение
            
            state["messages"].append("Ожидание ответа от студента")
            
            # Устанавливаем флаг ожидания ответа
            state["metadata"]["waiting_for_answer"] = True
            state["metadata"]["current_question_id"] = len(state["questions"])
            
            return state
            
        except Exception as e:
            error_msg = f"Ошибка в ожидании ответа: {str(e)}"
            state["errors"].append(error_msg)
            return state
    
    def _evaluate_answer_node(self, state: ExamState) -> ExamState:
        """Оценивает ответ студента"""
        try:
            # Проверяем, что есть вопрос для оценки
            if not state["questions"]:
                state["errors"].append("Нет вопроса для оценки ответа")
                return state
            
            # Получаем последний вопрос
            current_question = state["questions"][-1]
            
            # В реальной реализации ответ будет получен от пользователя
            # Для демонстрации используем placeholder
            student_answer = state["metadata"].get("current_answer", "Демонстрационный ответ студента")
            
            # Оценка ответа (EvaluationAgent видит полный текст ответа)
            evaluation_result = self.evaluation_agent.evaluate_answer(
                question=current_question['question'],
                student_answer=student_answer,
                key_points=current_question['key_points'],
                topic_level=current_question['topic_level'],
                detailed=True
            )
            
            if evaluation_result.get("error"):
                state["errors"].append(f"Ошибка оценки ответа: {evaluation_result['error']}")
                return state
            
            # Добавление метаданных для полной оценки
            evaluation_result.update({
                'answer': student_answer,  # Сохраняем для DiagnosticAgent
                'question_number': current_question['question_number'],
                'timestamp': datetime.now(),
                'question_metadata': {
                    'bloom_level': current_question.get('bloom_level'),
                    'question_type': current_question.get('question_type', 'unknown'),
                    'topic_level': current_question.get('topic_level')
                }
            })
            
            state["evaluations"].append(evaluation_result)
            
            # ВАЖНО: Создаем summary БЕЗ текста ответа для QuestionAgent
            evaluation_summary = self.evaluation_agent.create_evaluation_summary(
                evaluation_result, 
                current_question
            )
            state["evaluation_summaries"].append(evaluation_summary)
            
            state["messages"].append(f"Ответ оценен: {evaluation_result.get('total_score', 0)}/10")
            
            # Сбрасываем флаг ожидания
            state["metadata"]["waiting_for_answer"] = False
            
            return state
            
        except Exception as e:
            error_msg = f"Ошибка оценки ответа: {str(e)}"
            state["errors"].append(error_msg)
            return state
    
    def _update_progress_node(self, state: ExamState) -> ExamState:
        """Обновляет прогресс экзамена"""
        try:
            # Обновляем прогресс
            state = update_exam_progress(state)
            
            # Вычисляем статистику
            stats = calculate_exam_statistics(state)
            state["metadata"]["current_statistics"] = stats
            
            questions_answered = len(state["evaluations"])
            total_questions = state["max_questions"]
            
            state["messages"].append(f"Прогресс: {questions_answered}/{total_questions} вопросов")
            
            return state
            
        except Exception as e:
            error_msg = f"Ошибка обновления прогресса: {str(e)}"
            state["errors"].append(error_msg)
            return state
    
    def _decide_next_action(self, state: ExamState) -> str:
        """Решает, продолжить экзамен или завершить"""
        if state.get("errors"):
            return "complete"
        
        if should_continue_exam(state):
            return "continue"
        else:
            return "complete"
    
    def _check_continuation_node(self, state: ExamState) -> ExamState:
        """Проверяет, нужно ли продолжать экзамен или завершить его"""
        try:
            # Проверяем условия продолжения экзамена
            max_questions = state.get("max_questions", 5)
            current_question = state.get("current_question_number", 0)
            
            # Проверяем, достигли ли максимального количества вопросов
            if current_question >= max_questions:
                state["status"] = "ready_for_completion"
                state["messages"].append(f"Достигнуто максимальное количество вопросов ({max_questions})")
            else:
                # Проверяем другие условия остановки
                errors_count = len(state.get("errors", []))
                if errors_count > 3:  # Слишком много ошибок
                    state["status"] = "ready_for_completion"
                    state["messages"].append("Экзамен завершен из-за множественных ошибок")
                else:
                    state["status"] = "ready_for_next_question"
                    state["messages"].append("Экзамен может продолжаться")
            
            return state
            
        except Exception as e:
            error_msg = f"Ошибка проверки продолжения: {str(e)}"
            state["errors"].append(error_msg)
            state["status"] = "ready_for_completion"
            return state
    
    def _complete_exam_node(self, state: ExamState) -> ExamState:
        """Завершает экзамен"""
        try:
            state["status"] = "completed"
            state["end_time"] = datetime.now()
            
            # Вычисляем продолжительность
            if state["start_time"] and state["end_time"]:
                duration = state["end_time"] - state["start_time"]
                state["metadata"]["duration_seconds"] = duration.total_seconds()
                state["metadata"]["duration_minutes"] = duration.total_seconds() / 60
            
            state["messages"].append("Экзамен завершен")
            
            return state
            
        except Exception as e:
            error_msg = f"Ошибка завершения экзамена: {str(e)}"
            state["errors"].append(error_msg)
            return state
    
    def _run_diagnostics_node(self, state: ExamState) -> ExamState:
        """Запускает диагностику результатов"""
        try:
            if not state["evaluations"]:
                state["errors"].append("Нет оценок для диагностики")
                return state
            
            # Диагностика результатов
            diagnostic_result = self.diagnostic_agent.diagnose_exam_results(
                state["questions"],
                state["evaluations"]
            )
            
            if diagnostic_result.get("error"):
                state["errors"].append(f"Ошибка диагностики: {diagnostic_result['error']}")
                return state
            
            state["diagnostic_result"] = diagnostic_result
            
            state["messages"].append("Диагностика результатов завершена")
            
            return state
            
        except Exception as e:
            error_msg = f"Ошибка диагностики: {str(e)}"
            state["errors"].append(error_msg)
            return state
    
    def _finalize_results_node(self, state: ExamState) -> ExamState:
        """Финализирует результаты экзамена"""
        try:
            # Добавляем финальную статистику
            final_stats = calculate_exam_statistics(state)
            state["metadata"]["final_statistics"] = final_stats
            
            # Добавляем информацию о сессии в диагностический результат
            if state.get("diagnostic_result"):
                state["diagnostic_result"]["session_info"] = {
                    'session_id': state["session_id"],
                    'student_name': state["student_name"],
                    'duration': state["metadata"].get("duration_minutes", 0),
                    'questions_count': len(state["questions"]),
                    'completion_rate': len(state["evaluations"]) / len(state["questions"]) * 100 if state["questions"] else 0
                }
            
            # Сохраняем workflow в историю
            workflow_entry = {
                'session_id': state["session_id"],
                'completion_time': datetime.now(),
                'final_state': {
                    'status': state["status"],
                    'questions_count': len(state["questions"]),
                    'evaluations_count': len(state["evaluations"]),
                    'errors_count': len(state["errors"]),
                    'messages_count': len(state["messages"])
                }
            }
            self.workflow_history.append(workflow_entry)
            
            state["messages"].append("Результаты экзамена финализированы")
            
            return state
            
        except Exception as e:
            error_msg = f"Ошибка финализации результатов: {str(e)}"
            state["errors"].append(error_msg)
            return state
    
    def start_exam_workflow(self, student_name: str = "Студент") -> Dict[str, Any]:
        """
        Запускает полный workflow экзамена
        
        Args:
            student_name: Имя студента
            
        Returns:
            Результат выполнения workflow
        """
        try:
            # Создаем начальное состояние
            initial_state = create_initial_exam_state(
                student_name=student_name,
                subject=self.subject,
                topic_context=self.topic_context,
                difficulty=self.difficulty,
                max_questions=self.max_questions,
                use_theme_structure=self.use_theme_structure
            )
            
            # Запускаем workflow
            final_state = self.app.invoke(initial_state)
            
            return {
                'session_id': final_state['session_id'],
                'status': final_state['status'],
                'questions_count': len(final_state['questions']),
                'evaluations_count': len(final_state['evaluations']),
                'diagnostic_result': final_state.get('diagnostic_result'),
                'statistics': final_state['metadata'].get('final_statistics'),
                'errors': final_state['errors'],
                'messages': final_state['messages']
            }
            
        except Exception as e:
            return {
                'error': f"Критическая ошибка в workflow: {str(e)}",
                'session_id': None,
                'status': 'error'
            }
    
    def process_student_answer(self, session_state: ExamState, student_answer: str) -> ExamState:
        """
        Обрабатывает ответ студента в рамках текущей сессии
        
        Args:
            session_state: Текущее состояние сессии
            student_answer: Ответ студента
            
        Returns:
            Обновленное состояние
        """
        try:
            # Добавляем ответ в метаданные
            session_state["metadata"]["current_answer"] = student_answer
            
            # Продолжаем workflow с узла оценки ответа
            # В реальной реализации это будет более сложная логика
            updated_state = self._evaluate_answer_node(session_state)
            updated_state = self._update_progress_node(updated_state)
            
            return updated_state
            
        except Exception as e:
            session_state["errors"].append(f"Ошибка обработки ответа: {str(e)}")
            return session_state
    
    def get_workflow_status(self, session_id: str) -> Dict[str, Any]:
        """Возвращает статус workflow по ID сессии"""
        for entry in self.workflow_history:
            if entry['session_id'] == session_id:
                return entry
        
        return {'error': f'Сессия {session_id} не найдена'}
    
    def get_workflow_history(self) -> List[Dict]:
        """Возвращает историю workflow"""
        return self.workflow_history.copy()
    
    def reset_workflow(self):
        """Сбрасывает состояние workflow"""
        self.workflow_history = []
        
        # Сбрасываем историю агентов
        if hasattr(self.question_agent, 'reset_history'):
            self.question_agent.reset_history()
        if hasattr(self.evaluation_agent, 'reset_history'):
            self.evaluation_agent.reset_history()
        if hasattr(self.diagnostic_agent, 'reset_history'):
            self.diagnostic_agent.reset_history()
        if hasattr(self.theme_agent, 'reset_history'):
            self.theme_agent.reset_history()


# Функция для создания ExamWorkflow на LangGraph
def create_exam_workflow(
    topic_info: Dict[str, Any] = None,
    max_questions: int = 5,
    use_theme_structure: bool = False
) -> ExamWorkflowLangGraph:
    """Создает экземпляр ExamWorkflow на LangGraph"""
    return ExamWorkflowLangGraph(
        topic_info=topic_info,
        max_questions=max_questions,
        use_theme_structure=use_theme_structure
    )

# Псевдоним для обратной совместимости
def create_exam_workflow_langgraph(
    topic_info: Dict[str, Any] = None,
    max_questions: int = 5,
    use_theme_structure: bool = False
) -> ExamWorkflowLangGraph:
    """Создает экземпляр ExamWorkflow на LangGraph (псевдоним для обратной совместимости)"""
    return create_exam_workflow(topic_info, max_questions, use_theme_structure)


# Демонстрационная функция для тестирования workflow
def demo_exam_workflow():
    """Демонстрирует работу exam workflow"""
    print("🚀 Демонстрация ExamWorkflow на LangGraph")
    print("=" * 50)
    
    # Создаем workflow
    workflow = create_exam_workflow_langgraph(
        max_questions=3,
        use_theme_structure=True
    )
    
    # Запускаем экзамен
    print("Запуск экзамена...")
    result = workflow.start_exam_workflow("Демо Студент")
    
    # Выводим результаты
    print("\n📊 Результаты:")
    print(f"Session ID: {result.get('session_id')}")
    print(f"Статус: {result.get('status')}")
    print(f"Вопросов: {result.get('questions_count')}")
    print(f"Оценок: {result.get('evaluations_count')}")
    
    if result.get('errors'):
        print(f"\n❌ Ошибки: {result['errors']}")
    
    if result.get('messages'):
        print(f"\n📝 Сообщения:")
        for msg in result['messages']:
            print(f"  - {msg}")
    
    print("\n✅ Демонстрация завершена")


if __name__ == "__main__":
    demo_exam_workflow()
