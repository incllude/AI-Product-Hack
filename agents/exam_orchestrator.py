"""
Оркестратор экзамена на LangGraph - координирует работу всех агентов
"""
from typing import Dict, List, Optional, Any
from exam_workflow import create_exam_workflow_langgraph
from topic_manager import TopicManager
from base import ExamSession, create_initial_exam_state
import json
from datetime import datetime


class ExamOrchestratorLangGraph:
    """Координирует работу всех экзаменационных агентов на LangGraph"""
    
    def __init__(self, topic_info: Dict[str, Any] = None, max_questions: int = 5, use_theme_structure: bool = False):
        """
        Инициализация оркестратора
        
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
        
        # Создание workflow
        self.workflow = create_exam_workflow_langgraph(
            topic_info=topic_info,
            max_questions=max_questions,
            use_theme_structure=use_theme_structure
        )
        
        # Текущая сессия экзамена
        self.current_session = None
        self.session_history = []
    
    def start_exam(self, student_name: str = "Студент") -> Dict[str, Any]:
        """
        Начинает экзамен
        
        Args:
            student_name: Имя студента
            
        Returns:
            Информация о начале экзамена
        """
        try:
            # Создаем новую сессию
            self.current_session = ExamSession(
                student_name=student_name,
                subject=self.subject,
                difficulty=self.difficulty,
                topic_context=self.workflow.topic_context,
                max_questions=self.max_questions,
                use_theme_structure=self.use_theme_structure,
                start_time=datetime.now(),
                status="in_progress"
            )
            
            # Если используется тематическая структура, создаем её
            if self.use_theme_structure and self.workflow.theme_agent:
                theme_structure = self.workflow.theme_agent.generate_theme_structure(
                    total_questions=self.max_questions,
                    difficulty=self.difficulty
                )
                
                if not theme_structure.get("error"):
                    self.current_session.theme_structure = theme_structure
                    self.current_session.messages.append("Тематическая структура создана")
                else:
                    self.current_session.errors.append(f"Ошибка создания темы: {theme_structure['error']}")
            
            return {
                'session_id': self.current_session.session_id,
                'message': f"Экзамен начат для {student_name}",
                'subject': self.subject,
                'difficulty': self.difficulty,
                'max_questions': self.max_questions,
                'use_theme_structure': self.use_theme_structure,
                'status': 'started'
            }
            
        except Exception as e:
            return {
                'error': f"Ошибка начала экзамена: {str(e)}",
                'status': 'error'
            }
    
    def get_next_question(self) -> Dict[str, Any]:
        """
        Получает следующий вопрос от QuestionAgent
        
        Returns:
            Словарь с вопросом или сообщением об окончании
        """
        try:
            if not self.current_session:
                return {'error': 'Экзамен не начат'}
            
            if self.current_session.status != 'in_progress':
                return {'error': 'Экзамен не активен'}
            
            current_question_number = len(self.current_session.questions) + 1
            
            if current_question_number > self.max_questions:
                return {'message': 'Достигнуто максимальное количество вопросов'}
            
            # Получаем характеристики оценок БЕЗ текстов ответов для QuestionAgent
            evaluation_summaries = self.workflow.evaluation_agent.get_evaluation_summaries_for_question_agent()
            
            # Генерация вопроса
            question_data = self.workflow.question_agent.generate_question(
                current_question_number, 
                evaluation_summaries
            )
            
            if question_data.get("error"):
                self.current_session.errors.append(f"Ошибка генерации вопроса: {question_data['error']}")
                return {'error': question_data['error']}
            
            # Добавление метаданных
            question_data.update({
                'question_number': current_question_number,
                'timestamp': datetime.now(),
                'privacy_protected': True,
                'evaluation_summaries_count': len(evaluation_summaries),
                'data_flow': 'EvaluationAgent → characteristics → QuestionAgent'
            })
            
            self.current_session.questions.append(question_data)
            self.current_session.current_question_number = current_question_number
            
            return question_data
            
        except Exception as e:
            error_msg = f"Ошибка получения вопроса: {str(e)}"
            if self.current_session:
                self.current_session.errors.append(error_msg)
            return {'error': error_msg}
    
    def submit_answer(self, answer: str) -> Dict[str, Any]:
        """
        Отправляет ответ на оценку EvaluationAgent
        
        Args:
            answer: Ответ студента
            
        Returns:
            Результат оценки
        """
        try:
            if not self.current_session:
                return {'error': 'Экзамен не начат'}
            
            if self.current_session.status != 'in_progress':
                return {'error': 'Экзамен не активен'}
            
            if not self.current_session.questions:
                return {'error': 'Нет активного вопроса'}
            
            # Получаем последний вопрос
            current_question = self.current_session.questions[-1]
            
            # Проверяем, не отвечен ли уже этот вопрос
            if len(self.current_session.evaluations) >= len(self.current_session.questions):
                return {'error': 'На этот вопрос уже дан ответ'}
            
            # Оценка ответа (EvaluationAgent видит полный текст ответа)
            evaluation_result = self.workflow.evaluation_agent.evaluate_answer(
                question=current_question['question'],
                student_answer=answer,
                key_points=current_question['key_points'],
                topic_level=current_question['topic_level'],
                detailed=True
            )
            
            if evaluation_result.get("error"):
                self.current_session.errors.append(f"Ошибка оценки: {evaluation_result['error']}")
                return {'error': evaluation_result['error']}
            
            # Добавление метаданных для полной оценки
            evaluation_result.update({
                'answer': answer,  # Сохраняем для DiagnosticAgent
                'question_number': current_question['question_number'],
                'timestamp': datetime.now(),
                'question_metadata': {
                    'bloom_level': current_question.get('bloom_level'),
                    'question_type': current_question.get('question_type', 'unknown'),
                    'topic_level': current_question.get('topic_level')
                }
            })
            
            self.current_session.evaluations.append(evaluation_result)
            
            # Создаем summary БЕЗ текста ответа для QuestionAgent
            evaluation_summary = self.workflow.evaluation_agent.create_evaluation_summary(
                evaluation_result, 
                current_question
            )
            self.current_session.evaluation_summaries.append(evaluation_summary)
            
            # Обновляем прогресс
            self._update_session_progress()
            
            return evaluation_result
            
        except Exception as e:
            error_msg = f"Ошибка обработки ответа: {str(e)}"
            if self.current_session:
                self.current_session.errors.append(error_msg)
            return {'error': error_msg}
    
    def get_progress(self) -> Dict[str, Any]:
        """Возвращает информацию о прогрессе экзамена"""
        if not self.current_session:
            return {'error': 'Экзамен не начат'}
        
        questions_asked = len(self.current_session.questions)
        questions_answered = len(self.current_session.evaluations)
        
        total_score = sum(eval_data.get('total_score', 0) for eval_data in self.current_session.evaluations)
        max_possible_score = questions_answered * 10
        
        progress = {
            'session_id': self.current_session.session_id,
            'student_name': self.current_session.student_name,
            'questions_asked': questions_asked,
            'questions_answered': questions_answered,
            'max_questions': self.max_questions,
            'current_score': total_score,
            'max_possible_score': max_possible_score,
            'percentage': (total_score / max_possible_score * 100) if max_possible_score > 0 else 0,
            'status': self.current_session.status,
            'remaining_questions': max(0, self.max_questions - questions_asked)
        }
        
        # Добавляем информацию о тематической структуре, если используется
        if self.use_theme_structure and self.workflow.question_agent:
            theme_progress = self.workflow.question_agent.get_theme_progress()
            progress['theme_progress'] = theme_progress
        
        return progress
    
    def complete_exam(self) -> Dict[str, Any]:
        """
        Завершает экзамен и запускает DiagnosticAgent
        
        Returns:
            Результат диагностики
        """
        try:
            if not self.current_session:
                return {'error': 'Экзамен не начат'}
            
            if self.current_session.status != 'in_progress':
                return {'error': 'Экзамен уже завершен'}
            
            if not self.current_session.evaluations:
                return {'error': 'Нет ответов для анализа'}
            
            # Завершение экзамена
            self.current_session.end_time = datetime.now()
            self.current_session.status = 'completed'
            
            # Диагностика результатов
            diagnostic_result = self.workflow.diagnostic_agent.diagnose_exam_results(
                self.current_session.questions,
                self.current_session.evaluations
            )
            
            if diagnostic_result.get("error"):
                self.current_session.errors.append(f"Ошибка диагностики: {diagnostic_result['error']}")
                return {'error': diagnostic_result['error']}
            
            self.current_session.diagnostic_result = diagnostic_result
            
            # Добавление информации о сессии
            diagnostic_result['session_info'] = {
                'session_id': self.current_session.session_id,
                'student_name': self.current_session.student_name,
                'duration': self._calculate_duration(),
                'questions_count': len(self.current_session.questions),
                'completion_rate': len(self.current_session.evaluations) / len(self.current_session.questions) * 100
            }
            
            # Сохраняем сессию в историю
            self.session_history.append(self.current_session)
            
            return diagnostic_result
            
        except Exception as e:
            error_msg = f"Ошибка завершения экзамена: {str(e)}"
            if self.current_session:
                self.current_session.errors.append(error_msg)
            return {'error': error_msg}
    
    def _update_session_progress(self):
        """Обновляет прогресс текущей сессии"""
        if not self.current_session:
            return
        
        questions_count = len(self.current_session.questions)
        evaluations_count = len(self.current_session.evaluations)
        
        # Обновляем номер текущего вопроса
        self.current_session.current_question_number = questions_count
        
        # Обновляем метаданные
        self.current_session.metadata.update({
            'questions_asked': questions_count,
            'questions_answered': evaluations_count,
            'completion_percentage': (evaluations_count / self.max_questions * 100) if self.max_questions > 0 else 0,
            'last_updated': datetime.now()
        })
    
    def _calculate_duration(self) -> str:
        """Вычисляет продолжительность экзамена"""
        if not self.current_session or not self.current_session.start_time or not self.current_session.end_time:
            return "Неизвестно"
        
        duration = self.current_session.end_time - self.current_session.start_time
        minutes = duration.total_seconds() / 60
        
        if minutes < 1:
            return f"{int(duration.total_seconds())} сек"
        elif minutes < 60:
            return f"{int(minutes)} мин"
        else:
            hours = int(minutes // 60)
            remaining_minutes = int(minutes % 60)
            return f"{hours}ч {remaining_minutes}мин"
    
    def export_results(self, format: str = 'json') -> Dict[str, Any]:
        """
        Экспортирует результаты экзамена
        
        Args:
            format: Формат экспорта ('json', 'summary')
            
        Returns:
            Данные для экспорта
        """
        if not self.current_session:
            return {'error': 'Нет активной сессии для экспорта'}
        
        if format == 'summary':
            return self._create_summary()
        else:
            return {
                'session_info': {
                    'session_id': self.current_session.session_id,
                    'student_name': self.current_session.student_name,
                    'subject': self.current_session.subject,
                    'difficulty': self.current_session.difficulty,
                    'status': self.current_session.status,
                    'start_time': self.current_session.start_time.isoformat() if self.current_session.start_time else None,
                    'end_time': self.current_session.end_time.isoformat() if self.current_session.end_time else None,
                    'questions_count': len(self.current_session.questions),
                    'evaluations_count': len(self.current_session.evaluations)
                },
                'progress': self.get_progress(),
                'diagnostic_result': self.current_session.diagnostic_result,
                'export_timestamp': datetime.now().isoformat()
            }
    
    def _create_summary(self) -> Dict[str, Any]:
        """Создает краткую сводку результатов"""
        if not self.current_session:
            return {'error': 'Нет активной сессии'}
        
        progress = self.get_progress()
        
        summary = {
            'student': self.current_session.student_name,
            'subject': self.subject,
            'difficulty': self.difficulty,
            'duration': self._calculate_duration(),
            'total_score': f"{progress['current_score']}/{progress['max_possible_score']}",
            'percentage': f"{progress['percentage']:.1f}%",
            'questions_answered': f"{progress['questions_answered']}/{progress['questions_asked']}"
        }
        
        # Добавляем оценки по вопросам
        question_scores = []
        for i, eval_data in enumerate(self.current_session.evaluations, 1):
            question_scores.append({
                'question': i,
                'score': eval_data.get('total_score', 0),
                'question_text': self.current_session.questions[i-1]['question'][:100] + "..." 
                    if len(self.current_session.questions[i-1]['question']) > 100 
                    else self.current_session.questions[i-1]['question']
            })
        
        summary['detailed_scores'] = question_scores
        
        return summary
    
    def can_continue(self) -> bool:
        """Проверяет, можно ли продолжить экзамен"""
        return (
            self.current_session and
            self.current_session.status == 'in_progress' and 
            len(self.current_session.questions) < self.max_questions
        )
    
    def force_complete(self) -> Dict[str, Any]:
        """Принудительно завершает экзамен досрочно"""
        if self.current_session and self.current_session.status == 'in_progress':
            return self.complete_exam()
        else:
            return {'error': 'Экзамен не активен'}
    
    def get_session_info(self) -> Dict[str, Any]:
        """Возвращает информацию о текущей сессии"""
        if not self.current_session:
            return {'error': 'Нет активной сессии'}
        
        session_info = {
            'session_id': self.current_session.session_id,
            'student_name': self.current_session.student_name,
            'subject': self.subject,
            'difficulty': self.difficulty,
            'status': self.current_session.status,
            'start_time': self.current_session.start_time.isoformat() if self.current_session.start_time else None,
            'end_time': self.current_session.end_time.isoformat() if self.current_session.end_time else None,
            'questions_count': len(self.current_session.questions),
            'evaluations_count': len(self.current_session.evaluations),
            'use_theme_structure': self.use_theme_structure,
            'errors_count': len(self.current_session.errors),
            'messages_count': len(self.current_session.messages)
        }
        
        # Добавляем информацию о тематической структуре, если используется
        if self.use_theme_structure and self.workflow.question_agent:
            theme_progress = self.workflow.question_agent.get_theme_progress()
            session_info['theme_progress'] = theme_progress
        
        return session_info
    
    def get_theme_structure_info(self) -> Dict[str, Any]:
        """Возвращает информацию о тематической структуре экзамена"""
        if not self.use_theme_structure or not self.current_session or not self.current_session.theme_structure:
            return {'error': 'Тематическая структура не используется'}
        
        theme_structure = self.current_session.theme_structure
        
        return {
            'curriculum_id': theme_structure.get('curriculum_id'),
            'total_questions': theme_structure.get('total_questions'),
            'questions_distribution': theme_structure.get('questions_distribution'),
            'bloom_coverage': theme_structure.get('metadata', {}).get('bloom_coverage'),
            'estimated_duration': theme_structure.get('metadata', {}).get('estimated_duration'),
            'assessment_framework': theme_structure.get('assessment_framework')
        }
    
    def get_theme_summary_report(self) -> str:
        """Генерирует краткий отчет о тематической структуре экзамена"""
        if not self.use_theme_structure or not self.workflow.theme_agent or not self.current_session or not self.current_session.theme_structure:
            return "Тематическая структура не используется в данном экзамене."
        
        return self.workflow.theme_agent.generate_summary_report(self.current_session.theme_structure)
    
    def validate_theme_structure(self) -> Dict[str, Any]:
        """Валидирует тематическую структуру экзамена"""
        if not self.use_theme_structure or not self.workflow.theme_agent or not self.current_session or not self.current_session.theme_structure:
            return {'error': 'Тематическая структура не используется'}
        
        return self.workflow.theme_agent.validate_theme_structure(self.current_session.theme_structure)
    
    def export_theme_structure(self, format: str = 'json') -> str:
        """Экспортирует тематическую структуру экзамена"""
        if not self.use_theme_structure or not self.workflow.theme_agent or not self.current_session or not self.current_session.theme_structure:
            return '{"error": "Тематическая структура не используется"}'
        
        if format == 'json':
            return self.workflow.theme_agent.export_structure_to_json(self.current_session.theme_structure)
        else:
            return str(self.current_session.theme_structure)
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику workflow"""
        return {
            'workflow_history_count': len(self.workflow.get_workflow_history()),
            'session_history_count': len(self.session_history),
            'current_session_active': self.current_session is not None,
            'agents_statistics': {
                'question_agent': self.workflow.question_agent.get_agent_info(),
                'evaluation_agent': self.workflow.evaluation_agent.get_agent_info(),
                'diagnostic_agent': self.workflow.diagnostic_agent.get_agent_info(),
                'theme_agent': self.workflow.theme_agent.get_agent_info() if self.workflow.theme_agent else None
            }
        }
    
    def reset_orchestrator(self):
        """Сбрасывает состояние оркестратора"""
        self.current_session = None
        self.session_history = []
        self.workflow.reset_workflow()
    
    def get_session_history(self) -> List[Dict]:
        """Возвращает историю сессий"""
        return [
            {
                'session_id': session.session_id,
                'student_name': session.student_name,
                'status': session.status,
                'questions_count': len(session.questions),
                'evaluations_count': len(session.evaluations),
                'start_time': session.start_time.isoformat() if session.start_time else None,
                'end_time': session.end_time.isoformat() if session.end_time else None,
                'errors_count': len(session.errors)
            }
            for session in self.session_history
        ]


# Функция для создания ExamOrchestrator на LangGraph
def create_exam_orchestrator(
    topic_info: Dict[str, Any] = None,
    max_questions: int = 5,
    use_theme_structure: bool = False
) -> ExamOrchestratorLangGraph:
    """Создает экземпляр ExamOrchestrator на LangGraph"""
    return ExamOrchestratorLangGraph(
        topic_info=topic_info,
        max_questions=max_questions,
        use_theme_structure=use_theme_structure
    )

# Псевдоним для обратной совместимости
def create_exam_orchestrator_langgraph(
    topic_info: Dict[str, Any] = None,
    max_questions: int = 5,
    use_theme_structure: bool = False
) -> ExamOrchestratorLangGraph:
    """Создает экземпляр ExamOrchestrator на LangGraph (псевдоним для обратной совместимости)"""
    return create_exam_orchestrator(topic_info, max_questions, use_theme_structure)
