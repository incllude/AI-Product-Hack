"""
Оптимизированный оркестратор экзамена на LangGraph с быстрой инициализацией
"""
from typing import Dict, List, Optional, Any
from exam_workflow_optimized import create_optimized_exam_workflow, SharedLLMManager
from topic_manager import TopicManager
from base import ExamSession, create_initial_exam_state
import json
from datetime import datetime


class OptimizedExamOrchestratorLangGraph:
    """Оптимизированный координатор работы всех экзаменационных агентов на LangGraph"""
    
    def __init__(self, topic_info: Dict[str, Any] = None, max_questions: int = 5, use_theme_structure: bool = False):
        """
        Оптимизированная инициализация оркестратора
        
        Args:
            topic_info: Информация о теме экзамена (из TopicManager)
            max_questions: Максимальное количество вопросов
            use_theme_structure: Использовать ли тематическую структуру по таксономии Блума
        """
        print("🚀 Оптимизированная инициализация оркестратора...")
        start_time = datetime.now()
        
        # Если тема не указана, используем тему по умолчанию
        if not topic_info:
            topic_manager = TopicManager()
            topic_info = topic_manager._get_default_topic()
        
        self.topic_info = topic_info
        self.difficulty = topic_info['difficulty']
        self.max_questions = max_questions
        self.use_theme_structure = use_theme_structure
        
        # Создание оптимизированного workflow (быстрая инициализация)
        self.workflow = create_optimized_exam_workflow(
            topic_info=topic_info,
            max_questions=max_questions,
            use_theme_structure=use_theme_structure
        )
        
        # Текущая сессия экзамена
        self.current_session = None
        self.session_history = []
        
        end_time = datetime.now()
        init_duration = (end_time - start_time).total_seconds()
        print(f"✅ Оптимизированный оркестратор инициализирован за {init_duration:.2f} секунд")
    
    def start_exam(self, student_name: str = "Студент") -> Dict[str, Any]:
        """
        Начинает экзамен с оптимизированной инициализацией
        
        Args:
            student_name: Имя студента
            
        Returns:
            Информация о начале экзамена
        """
        try:
            print("🎓 Запуск экзамена...")
            start_time = datetime.now()
            
            # Создаем новую сессию
            self.current_session = ExamSession(
                student_name=student_name,
                subject="Общие знания",  # Default value since self.subject is removed
                difficulty=self.difficulty,
                topic_context=self.workflow.topic_context,
                max_questions=self.max_questions,
                use_theme_structure=self.use_theme_structure,
                start_time=datetime.now(),
                status="in_progress"
            )
            
            # Инициализируем агенты только если используется тематическая структура
            if self.use_theme_structure and self.workflow.theme_agent:
                print("🔧 Создание тематической структуры...")
                theme_structure = self.workflow.theme_agent.generate_theme_structure(
                    total_questions=self.max_questions,
                    difficulty=self.difficulty
                )
                
                if not theme_structure.get("error"):
                    self.current_session.theme_structure = theme_structure
                    self.current_session.messages.append("Тематическая структура создана")
                else:
                    self.current_session.errors.append(f"Ошибка создания темы: {theme_structure['error']}")
            
            end_time = datetime.now()
            start_duration = (end_time - start_time).total_seconds()
            print(f"✅ Экзамен запущен за {start_duration:.2f} секунд")
            
            return {
                'session_id': self.current_session.session_id,
                'student_name': student_name,
                'subject': "Общие знания",  # Default value since self.subject is removed
                'difficulty': self.difficulty,
                'max_questions': self.max_questions,
                'use_theme_structure': self.use_theme_structure,
                'start_time': self.current_session.start_time.isoformat(),
                'status': 'started',
                'performance': {
                    'start_duration_seconds': start_duration,
                    'workflow_stats': self.workflow.get_performance_stats()
                }
            }
            
        except Exception as e:
            error_msg = f"Ошибка запуска экзамена: {str(e)}"
            if self.current_session:
                self.current_session.errors.append(error_msg)
            return {'error': error_msg}
    
    def get_next_question(self) -> Dict[str, Any]:
        """
        Получает следующий вопрос от QuestionAgent с ленивой инициализацией
        
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
            
            print(f"❓ Генерация вопроса {current_question_number}...")
            start_time = datetime.now()
            
            # Получаем характеристики оценок БЕЗ текстов ответов для QuestionAgent
            evaluation_summaries = self.workflow.evaluation_agent.get_evaluation_summaries_for_question_agent()
            
            # Генерация вопроса (агент инициализируется автоматически при первом обращении)
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
            
            end_time = datetime.now()
            question_duration = (end_time - start_time).total_seconds()
            print(f"✅ Вопрос {current_question_number} сгенерирован за {question_duration:.2f} секунд")
            
            # Добавляем информацию о производительности
            question_data['performance'] = {
                'generation_duration_seconds': question_duration,
                'agent_initialized': self.workflow._agents_initialized
            }
            
            return question_data
            
        except Exception as e:
            error_msg = f"Ошибка получения вопроса: {str(e)}"
            if self.current_session:
                self.current_session.errors.append(error_msg)
            return {'error': error_msg}
    
    def submit_answer(self, answer: str) -> Dict[str, Any]:
        """
        Отправляет ответ на оценку EvaluationAgent с оптимизацией
        
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
            
            print("📊 Оценка ответа...")
            start_time = datetime.now()
            
            # Оценка ответа (EvaluationAgent инициализируется автоматически при первом обращении)
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
            
            # Добавляем метаданные
            evaluation_result.update({
                'question_number': current_question['question_number'],
                'timestamp': datetime.now(),
                'student_answer_length': len(answer),
                'question_metadata': {
                    'topic_level': current_question.get('topic_level'),
                    'question_type': current_question.get('question_type')
                }
            })
            
            self.current_session.evaluations.append(evaluation_result)
            
            end_time = datetime.now()
            evaluation_duration = (end_time - start_time).total_seconds()
            print(f"✅ Ответ оценен за {evaluation_duration:.2f} секунд")
            
            # Добавляем информацию о производительности
            evaluation_result['performance'] = {
                'evaluation_duration_seconds': evaluation_duration,
                'shared_llm_stats': self.workflow.get_performance_stats()['shared_llm_stats']
            }
            
            return evaluation_result
            
        except Exception as e:
            error_msg = f"Ошибка оценки ответа: {str(e)}"
            if self.current_session:
                self.current_session.errors.append(error_msg)
            return {'error': error_msg}
    
    def complete_exam(self) -> Dict[str, Any]:
        """
        Завершает экзамен и генерирует финальный отчет с оптимизацией
        
        Returns:
            Финальный отчет экзамена
        """
        try:
            if not self.current_session:
                return {'error': 'Экзамен не начат'}
            
            print("📝 Генерация финального отчета...")
            start_time = datetime.now()
            
            # Завершаем сессию
            if self.current_session.status == 'in_progress':
                self.current_session.status = 'completed'
                self.current_session.end_time = datetime.now()
            
            # Запускаем диагностику (DiagnosticAgent инициализируется автоматически)
            diagnostic_result = self.workflow.diagnostic_agent.run_diagnostics(
                self.current_session.questions,
                self.current_session.evaluations
            )
            
            if diagnostic_result.get("error"):
                self.current_session.errors.append(f"Ошибка диагностики: {diagnostic_result['error']}")
                return {'error': diagnostic_result['error']}
            
            # Добавляем диагностический результат в сессию
            self.current_session.diagnostic_result = diagnostic_result
            
            # Добавляем сессию в историю
            self.session_history.append(self.current_session)
            
            end_time = datetime.now()
            completion_duration = (end_time - start_time).total_seconds()
            print(f"✅ Финальный отчет сгенерирован за {completion_duration:.2f} секунд")
            
            # Формируем финальный отчет с информацией о производительности
            final_report = {
                'session_id': self.current_session.session_id,
                'student_name': self.current_session.student_name,
                'exam_duration': (self.current_session.end_time - self.current_session.start_time).total_seconds(),
                'questions_count': len(self.current_session.questions),
                'evaluations_count': len(self.current_session.evaluations),
                'grade_info': diagnostic_result.get('grade_info', {}),
                'recommendations': diagnostic_result.get('recommendations', []),
                'statistics': diagnostic_result.get('statistics', {}),
                'performance': {
                    'completion_duration_seconds': completion_duration,
                    'total_exam_duration_seconds': (self.current_session.end_time - self.current_session.start_time).total_seconds(),
                    'workflow_performance': self.workflow.get_performance_stats(),
                    'errors_count': len(self.current_session.errors)
                }
            }
            
            return final_report
            
        except Exception as e:
            error_msg = f"Ошибка завершения экзамена: {str(e)}"
            if self.current_session:
                self.current_session.errors.append(error_msg)
            return {'error': error_msg}
    
    def get_progress(self) -> Dict[str, Any]:
        """Возвращает прогресс экзамена с информацией о производительности"""
        if not self.current_session:
            return {
                'status': 'not_started',
                'questions_answered': 0,
                'max_questions': self.max_questions,
                'current_score': 0,
                'max_possible_score': 0,
                'performance': {
                    'agents_initialized': False,
                    'shared_llm_available': SharedLLMManager._llm is not None
                }
            }
        
        current_score = sum(eval_data.get('total_score', 0) for eval_data in self.current_session.evaluations)
        max_possible_score = len(self.current_session.evaluations) * 10
        
        return {
            'status': self.current_session.status,
            'questions_answered': len(self.current_session.evaluations),
            'total_questions': len(self.current_session.questions),
            'max_questions': self.max_questions,
            'current_score': current_score,
            'max_possible_score': max_possible_score,
            'errors_count': len(self.current_session.errors),
            'performance': {
                'agents_initialized': self.workflow._agents_initialized,
                'workflow_stats': self.workflow.get_performance_stats(),
                'session_duration': (datetime.now() - self.current_session.start_time).total_seconds() if self.current_session.start_time else 0
            }
        }
    
    def can_continue(self) -> bool:
        """Проверяет, можно ли продолжить экзамен"""
        if not self.current_session:
            return False
        
        if self.current_session.status != 'in_progress':
            return False
        
        # Проверяем, не превышено ли максимальное количество вопросов
        questions_asked = len(self.current_session.questions)
        answers_given = len(self.current_session.evaluations)
        
        # Можем продолжить, если:
        # 1. Не достигли максимального количества вопросов
        # 2. На все заданные вопросы даны ответы
        return questions_asked < self.max_questions and questions_asked == answers_given
    
    def force_complete(self) -> Dict[str, Any]:
        """Принудительно завершает экзамен"""
        if self.current_session and self.current_session.status == 'in_progress':
            self.current_session.status = 'force_completed'
            self.current_session.end_time = datetime.now()
            self.current_session.messages.append("Экзамен принудительно завершен")
        
        return self.complete_exam()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Получение сводки производительности системы"""
        return {
            'optimization_enabled': True,
            'shared_llm_manager': {
                'initialized': SharedLLMManager._llm is not None,
                'stats': SharedLLMManager._llm.__dict__ if SharedLLMManager._llm else None
            },
            'workflow_performance': self.workflow.get_performance_stats(),
            'current_session': {
                'active': self.current_session is not None,
                'status': self.current_session.status if self.current_session else None,
                'questions': len(self.current_session.questions) if self.current_session else 0,
                'evaluations': len(self.current_session.evaluations) if self.current_session else 0,
                'errors': len(self.current_session.errors) if self.current_session else 0
            },
            'session_history_count': len(self.session_history)
        }


# Функция для создания оптимизированного оркестратора
def create_optimized_exam_orchestrator(
    topic_info: Dict[str, Any] = None,
    max_questions: int = 5,
    use_theme_structure: bool = False
) -> OptimizedExamOrchestratorLangGraph:
    """Создает экземпляр оптимизированного ExamOrchestrator на LangGraph"""
    return OptimizedExamOrchestratorLangGraph(
        topic_info=topic_info,
        max_questions=max_questions,
        use_theme_structure=use_theme_structure
    )

# Alias для обратной совместимости с фронтендом
OptimizedExamOrchestrator = OptimizedExamOrchestratorLangGraph
