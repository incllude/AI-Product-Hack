"""
Базовые классы и состояния для LangGraph агентов
"""
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from operator import add

# Состояния для LangGraph workflow
class ExamState(TypedDict):
    """Основное состояние экзамена"""
    # Основная информация о сессии
    session_id: str
    student_name: str
    subject: str
    difficulty: str
    topic_context: str
    status: str  # 'not_started', 'in_progress', 'completed'
    
    # Временные метки
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    
    # Настройки экзамена
    max_questions: int
    use_theme_structure: bool
    current_question_number: int
    
    # Тематическая структура
    theme_structure: Optional[Dict[str, Any]]
    theme_progress: Optional[Dict[str, Any]]
    
    # Данные вопросов и ответов
    questions: Annotated[List[Dict], add]
    evaluations: Annotated[List[Dict], add]
    evaluation_summaries: Annotated[List[Dict], add]
    
    # Результаты диагностики
    diagnostic_result: Optional[Dict[str, Any]]
    
    # Сообщения и ошибки
    messages: Annotated[List[str], add]
    errors: Annotated[List[str], add]
    
    # Метаданные
    metadata: Dict[str, Any]

class QuestionState(TypedDict):
    """Состояние для генерации вопросов"""
    question_number: int
    evaluation_summaries: List[Dict]
    theme_requirements: Optional[Dict]
    generated_question: Optional[Dict]
    error: Optional[str]
    # Добавленные поля для LangGraph workflow
    raw_response: Optional[str]        # Сырой ответ от LLM
    question_type: Optional[str]       # Тип генерации: "initial", "contextual", "theme_guided"

class EvaluationState(TypedDict):
    """Состояние для оценки ответов"""
    question: str
    student_answer: str
    key_points: str
    topic_level: str
    question_metadata: Dict[str, Any]
    evaluation_result: Optional[Dict]
    evaluation_summary: Optional[Dict]
    error: Optional[str]
    # Добавленные поля для LangGraph workflow
    evaluation_type: Optional[str]     # Тип оценки: "detailed", "empty"
    raw_evaluation: Optional[str]      # Сырой ответ от LLM

class DiagnosticState(TypedDict):
    """Состояние для диагностики"""
    questions: List[Dict]
    evaluations: List[Dict]
    diagnostic_result: Optional[Dict]
    error: Optional[str]
    # Добавленные поля для LangGraph workflow
    analysis_data: Optional[str]           # Подготовленные данные для анализа
    pattern_analysis: Optional[Dict]       # Результат анализа паттернов
    statistics: Optional[Dict]             # Вычисленная статистика
    grade_info: Optional[Dict]             # Информация об оценке и градации
    final_report: Optional[str]            # Финальный отчет
    recommendations: Optional[List[str]]   # Список рекомендаций
    critical_areas: Optional[List[str]]    # Критические области для улучшения

class ThemeState(TypedDict):
    """Состояние для тематической структуры"""
    subject: str
    topic_context: str
    total_questions: int
    difficulty: str
    theme_structure: Optional[Dict]
    validation_result: Optional[Dict]
    error: Optional[str]
    # Добавленные поля для LangGraph workflow
    bloom_levels_info: Optional[Dict]      # Информация об уровнях Блума
    raw_theme_structure: Optional[str]     # Сырая тематическая структура от LLM
    questions_distribution: Optional[Dict] # Распределение вопросов по уровням
    question_guidelines: Optional[Dict]    # Руководящие принципы для генерации вопросов

@dataclass
class ExamSession:
    """Класс для управления сессией экзамена"""
    session_id: str = field(default_factory=lambda: f"exam_{uuid.uuid4().hex[:8]}")
    student_name: str = "Студент"
    subject: str = "Общие знания"
    difficulty: str = "средний"
    topic_context: str = ""
    status: str = "not_started"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_questions: int = 5
    use_theme_structure: bool = False
    current_question_number: int = 0
    theme_structure: Optional[Dict] = None
    theme_progress: Optional[Dict] = None
    questions: List[Dict] = field(default_factory=list)
    evaluations: List[Dict] = field(default_factory=list)
    evaluation_summaries: List[Dict] = field(default_factory=list)
    diagnostic_result: Optional[Dict] = None
    messages: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_state(self) -> ExamState:
        """Конвертирует в ExamState для LangGraph"""
        return ExamState(
            session_id=self.session_id,
            student_name=self.student_name,
            subject=self.subject,
            difficulty=self.difficulty,
            topic_context=self.topic_context,
            status=self.status,
            start_time=self.start_time,
            end_time=self.end_time,
            max_questions=self.max_questions,
            use_theme_structure=self.use_theme_structure,
            current_question_number=self.current_question_number,
            theme_structure=self.theme_structure,
            theme_progress=self.theme_progress,
            questions=self.questions,
            evaluations=self.evaluations,
            evaluation_summaries=self.evaluation_summaries,
            diagnostic_result=self.diagnostic_result,
            messages=self.messages,
            errors=self.errors,
            metadata=self.metadata
        )
    
    @classmethod
    def from_state(cls, state: ExamState) -> 'ExamSession':
        """Создает ExamSession из ExamState"""
        return cls(
            session_id=state['session_id'],
            student_name=state['student_name'],
            subject=state['subject'],
            difficulty=state['difficulty'],
            topic_context=state['topic_context'],
            status=state['status'],
            start_time=state['start_time'],
            end_time=state['end_time'],
            max_questions=state['max_questions'],
            use_theme_structure=state['use_theme_structure'],
            current_question_number=state['current_question_number'],
            theme_structure=state['theme_structure'],
            theme_progress=state['theme_progress'],
            questions=state['questions'],
            evaluations=state['evaluations'],
            evaluation_summaries=state['evaluation_summaries'],
            diagnostic_result=state['diagnostic_result'],
            messages=state['messages'],
            errors=state['errors'],
            metadata=state['metadata']
        )

class LangGraphAgentBase:
    """Базовый класс для всех LangGraph агентов"""
    
    def __init__(self, subject: str = "Общие знания", topic_context: str = None):
        self.subject = subject
        self.topic_context = topic_context or f"Общий экзамен по предмету {subject}"
        self.agent_id = f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        self.history = []
    
    def log_operation(self, operation: str, input_data: Any, output_data: Any, error: Optional[str] = None):
        """Логирует операцию агента"""
        log_entry = {
            'timestamp': datetime.now(),
            'agent_id': self.agent_id,
            'operation': operation,
            'input_data': str(input_data)[:200],  # Ограничиваем размер лога
            'output_data': str(output_data)[:200],
            'error': error,
            'success': error is None
        }
        self.history.append(log_entry)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Возвращает информацию об агенте"""
        return {
            'agent_class': self.__class__.__name__,
            'agent_id': self.agent_id,
            'subject': self.subject,
            'topic_context': self.topic_context[:100] + "..." if len(self.topic_context) > 100 else self.topic_context,
            'operations_count': len(self.history),
            'last_operation': self.history[-1] if self.history else None
        }
    
    def reset_history(self):
        """Очищает историю операций"""
        self.history = []

def create_initial_exam_state(
    student_name: str = "Студент",
    subject: str = "Общие знания", 
    topic_context: str = "",
    difficulty: str = "средний",
    max_questions: int = 5,
    use_theme_structure: bool = False
) -> ExamState:
    """Создает начальное состояние экзамена"""
    session_id = f"exam_{uuid.uuid4().hex[:8]}"
    
    return ExamState(
        session_id=session_id,
        student_name=student_name,
        subject=subject,
        difficulty=difficulty,
        topic_context=topic_context,
        status="not_started",
        start_time=None,
        end_time=None,
        max_questions=max_questions,
        use_theme_structure=use_theme_structure,
        current_question_number=0,
        theme_structure=None,
        theme_progress=None,
        questions=[],
        evaluations=[],
        evaluation_summaries=[],
        diagnostic_result=None,
        messages=[],
        errors=[],
        metadata={}
    )

def validate_exam_state(state: ExamState) -> List[str]:
    """Валидирует состояние экзамена"""
    errors = []
    
    # Проверка обязательных полей
    required_fields = ['session_id', 'student_name', 'subject', 'status']
    for field in required_fields:
        if not state.get(field):
            errors.append(f"Отсутствует обязательное поле: {field}")
    
    # Проверка статуса
    valid_statuses = ['not_started', 'in_progress', 'completed']
    if state.get('status') not in valid_statuses:
        errors.append(f"Неверный статус: {state.get('status')}")
    
    # Проверка номера текущего вопроса
    if state.get('current_question_number', 0) < 0:
        errors.append("Номер вопроса не может быть отрицательным")
    
    if state.get('current_question_number', 0) > state.get('max_questions', 5):
        errors.append("Номер вопроса превышает максимальное количество")
    
    # Проверка согласованности данных
    questions_count = len(state.get('questions', []))
    evaluations_count = len(state.get('evaluations', []))
    
    if evaluations_count > questions_count:
        errors.append("Количество оценок превышает количество вопросов")
    
    return errors

def update_exam_progress(state: ExamState) -> ExamState:
    """Обновляет прогресс экзамена"""
    questions_count = len(state.get('questions', []))
    evaluations_count = len(state.get('evaluations', []))
    
    # Обновляем номер текущего вопроса
    state['current_question_number'] = questions_count
    
    # Обновляем метаданные
    if 'metadata' not in state:
        state['metadata'] = {}
    
    state['metadata'].update({
        'questions_asked': questions_count,
        'questions_answered': evaluations_count,
        'completion_percentage': (evaluations_count / state['max_questions'] * 100) if state['max_questions'] > 0 else 0,
        'last_updated': datetime.now()
    })
    
    return state

def should_continue_exam(state: ExamState) -> bool:
    """Проверяет, можно ли продолжить экзамен"""
    return (
        state.get('status') == 'in_progress' and
        len(state.get('questions', [])) < state.get('max_questions', 5) and
        not state.get('errors')
    )

def calculate_exam_statistics(state: ExamState) -> Dict[str, Any]:
    """Вычисляет статистику экзамена"""
    evaluations = state.get('evaluations', [])
    
    if not evaluations:
        return {'message': 'Нет данных для статистики'}
    
    scores = [eval_data.get('total_score', 0) for eval_data in evaluations]
    total_score = sum(scores)
    max_possible = len(scores) * 10
    
    return {
        'total_score': total_score,
        'max_possible_score': max_possible,
        'average_score': total_score / len(scores) if scores else 0,
        'percentage': (total_score / max_possible * 100) if max_possible > 0 else 0,
        'highest_score': max(scores) if scores else 0,
        'lowest_score': min(scores) if scores else 0,
        'questions_answered': len(evaluations),
        'completion_rate': len(evaluations) / state.get('max_questions', 5) * 100
    }
