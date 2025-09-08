"""
Оркестратор экзамена - координирует работу всех агентов
"""
from typing import Dict, List, Optional, Tuple
from question_agent import QuestionAgent
from evaluation_agent import EvaluationAgent
from diagnostic_agent import DiagnosticAgent
from topic_manager import TopicManager
from theme_agent import ThemeAgent
import json
import time
from datetime import datetime


class ExamOrchestrator:
    """Координирует работу всех экзаменационных агентов"""
    
    def __init__(self, topic_info: Dict[str, any] = None, max_questions: int = 5, use_theme_structure: bool = False):
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
        
        # Создание контекста для агентов
        topic_manager = TopicManager()
        topic_context = topic_manager.get_topic_context_for_prompts(topic_info)
        
        # Создание тематического агента (если нужно)
        self.theme_agent = None
        self.theme_structure = None
        
        if use_theme_structure:
            self.theme_agent = ThemeAgent(
                subject=self.subject,
                topic_context=topic_context
            )
            # Генерируем тематическую структуру экзамена
            self.theme_structure = self.theme_agent.generate_theme_structure(
                total_questions=max_questions,
                difficulty=self.difficulty
            )
        
        # Создание агентов с контекстом темы и тематической структурой
        self.question_agent = QuestionAgent(
            subject=self.subject, 
            difficulty=self.difficulty,
            topic_context=topic_context,
            theme_structure=self.theme_structure
        )
        self.evaluation_agent = EvaluationAgent(
            subject=self.subject,
            topic_context=topic_context
        )
        self.diagnostic_agent = DiagnosticAgent(
            subject=self.subject,
            topic_context=topic_context
        )
        
        # Данные экзамена
        self.exam_session = {
            'session_id': self._generate_session_id(),
            'topic_info': topic_info,
            'subject': self.subject,
            'difficulty': self.difficulty,
            'use_theme_structure': use_theme_structure,
            'theme_structure': self.theme_structure,
            'start_time': None,
            'end_time': None,
            'questions': [],
            'evaluations': [],
            'student_name': None,
            'status': 'not_started'  # not_started, in_progress, completed
        }
    
    def _generate_session_id(self) -> str:
        """Генерирует уникальный ID сессии"""
        return f"exam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def start_exam(self, student_name: str = "Студент") -> Dict[str, any]:
        """
        Начинает экзамен
        
        Args:
            student_name: Имя студента
            
        Returns:
            Информация о начале экзамена
        """
        self.exam_session['student_name'] = student_name
        self.exam_session['start_time'] = datetime.now()
        self.exam_session['status'] = 'in_progress'
        
        return {
            'session_id': self.exam_session['session_id'],
            'message': f"Экзамен начат для {student_name}",
            'subject': self.subject,
            'difficulty': self.difficulty,
            'max_questions': self.max_questions
        }
    
    def get_next_question(self) -> Dict[str, any]:
        """
        Получает следующий вопрос от QuestionAgent на основе характеристик от EvaluationAgent
        
        Returns:
            Словарь с вопросом или сообщением об окончании
        """
        if self.exam_session['status'] != 'in_progress':
            return {'error': 'Экзамен не начат или уже завершен'}
        
        current_question_number = len(self.exam_session['questions']) + 1
        
        if current_question_number > self.max_questions:
            return {'message': 'Достигнуто максимальное количество вопросов'}
        
        # Получаем характеристики оценок БЕЗ текстов ответов для QuestionAgent
        evaluation_summaries = self.evaluation_agent.get_evaluation_summaries_for_question_agent()
        
        # Генерация вопроса на основе характеристик (НЕ текстов ответов)
        question_data = self.question_agent.generate_question(
            current_question_number, 
            evaluation_summaries  # Только характеристики, БЕЗ текстов ответов
        )
        
        # Добавление метаданных о приватности
        question_data['question_number'] = current_question_number
        question_data['timestamp'] = datetime.now()
        question_data['privacy_protected'] = True  # Подтверждение соблюдения приватности
        question_data['evaluation_summaries_count'] = len(evaluation_summaries)
        question_data['data_flow'] = 'EvaluationAgent → characteristics → QuestionAgent'
        
        self.exam_session['questions'].append(question_data)
        
        return question_data
    
    def submit_answer(self, answer: str) -> Dict[str, any]:
        """
        Отправляет ответ на оценку EvaluationAgent
        
        Args:
            answer: Ответ студента
            
        Returns:
            Результат оценки (БЕЗ передачи текста ответа в QuestionAgent)
        """
        if self.exam_session['status'] != 'in_progress':
            return {'error': 'Экзамен не начат или уже завершен'}
        
        if not self.exam_session['questions']:
            return {'error': 'Нет активного вопроса'}
        
        # Получаем последний вопрос
        current_question = self.exam_session['questions'][-1]
        
        # Проверяем, не отвечен ли уже этот вопрос
        if len(self.exam_session['evaluations']) >= len(self.exam_session['questions']):
            return {'error': 'На этот вопрос уже дан ответ'}
        
        # Оценка ответа (EvaluationAgent видит полный текст ответа)
        evaluation_result = self.evaluation_agent.evaluate_answer(
            question=current_question['question'],
            student_answer=answer,
            key_points=current_question['key_points'],
            topic_level=current_question['topic_level'],
            detailed=True
        )
        
        # Добавление метаданных для полной оценки
        evaluation_result['answer'] = answer  # Сохраняем для DiagnosticAgent
        evaluation_result['question_number'] = current_question['question_number']
        evaluation_result['timestamp'] = datetime.now()
        evaluation_result['question_metadata'] = {
            'bloom_level': current_question.get('bloom_level'),
            'question_type': current_question.get('question_type', 'unknown'),
            'topic_level': current_question.get('topic_level')
        }
        
        self.exam_session['evaluations'].append(evaluation_result)
        
        # ВАЖНО: Возвращаем полную оценку, но QuestionAgent получит только summary
        return evaluation_result
    
    def get_progress(self) -> Dict[str, any]:
        """Возвращает информацию о прогрессе экзамена"""
        questions_asked = len(self.exam_session['questions'])
        questions_answered = len(self.exam_session['evaluations'])
        
        total_score = sum(eval_data.get('total_score', 0) for eval_data in self.exam_session['evaluations'])
        max_possible_score = questions_answered * 10
        
        return {
            'session_id': self.exam_session['session_id'],
            'student_name': self.exam_session['student_name'],
            'questions_asked': questions_asked,
            'questions_answered': questions_answered,
            'max_questions': self.max_questions,
            'current_score': total_score,
            'max_possible_score': max_possible_score,
            'percentage': (total_score / max_possible_score * 100) if max_possible_score > 0 else 0,
            'status': self.exam_session['status'],
            'remaining_questions': max(0, self.max_questions - questions_asked)
        }
    
    def complete_exam(self) -> Dict[str, any]:
        """
        Завершает экзамен и запускает DiagnosticAgent
        
        Returns:
            Результат диагностики
        """
        if self.exam_session['status'] != 'in_progress':
            return {'error': 'Экзамен не начат или уже завершен'}
        
        if not self.exam_session['evaluations']:
            return {'error': 'Нет ответов для анализа'}
        
        # Завершение экзамена
        self.exam_session['end_time'] = datetime.now()
        self.exam_session['status'] = 'completed'
        
        # Диагностика результатов
        diagnostic_result = self.diagnostic_agent.diagnose_exam_results(
            self.exam_session['questions'],
            self.exam_session['evaluations']
        )
        
        # Добавление информации о сессии
        diagnostic_result['session_info'] = {
            'session_id': self.exam_session['session_id'],
            'student_name': self.exam_session['student_name'],
            'duration': self._calculate_duration(),
            'questions_count': len(self.exam_session['questions']),
            'completion_rate': len(self.exam_session['evaluations']) / len(self.exam_session['questions']) * 100
        }
        
        return diagnostic_result
    
    def _calculate_duration(self) -> str:
        """Вычисляет продолжительность экзамена"""
        if not self.exam_session['start_time'] or not self.exam_session['end_time']:
            return "Неизвестно"
        
        duration = self.exam_session['end_time'] - self.exam_session['start_time']
        minutes = duration.total_seconds() / 60
        
        if minutes < 1:
            return f"{int(duration.total_seconds())} сек"
        elif minutes < 60:
            return f"{int(minutes)} мин"
        else:
            hours = int(minutes // 60)
            remaining_minutes = int(minutes % 60)
            return f"{hours}ч {remaining_minutes}мин"
    
    def export_results(self, format: str = 'json') -> Dict[str, any]:
        """
        Экспортирует результаты экзамена
        
        Args:
            format: Формат экспорта ('json', 'summary')
            
        Returns:
            Данные для экспорта
        """
        if format == 'summary':
            return self._create_summary()
        else:
            return {
                'session_info': self.exam_session,
                'progress': self.get_progress(),
                'export_timestamp': datetime.now().isoformat()
            }
    
    def _create_summary(self) -> Dict[str, any]:
        """Создает краткую сводку результатов"""
        progress = self.get_progress()
        
        summary = {
            'student': self.exam_session['student_name'],
            'subject': self.subject,
            'difficulty': self.difficulty,
            'duration': self._calculate_duration(),
            'total_score': f"{progress['current_score']}/{progress['max_possible_score']}",
            'percentage': f"{progress['percentage']:.1f}%",
            'questions_answered': f"{progress['questions_answered']}/{progress['questions_asked']}"
        }
        
        # Добавляем оценки по вопросам
        question_scores = []
        for i, eval_data in enumerate(self.exam_session['evaluations'], 1):
            question_scores.append({
                'question': i,
                'score': eval_data.get('total_score', 0),
                'question_text': self.exam_session['questions'][i-1]['question'][:100] + "..." if len(self.exam_session['questions'][i-1]['question']) > 100 else self.exam_session['questions'][i-1]['question']
            })
        
        summary['detailed_scores'] = question_scores
        
        return summary
    
    def can_continue(self) -> bool:
        """Проверяет, можно ли продолжить экзамен"""
        return (self.exam_session['status'] == 'in_progress' and 
                len(self.exam_session['questions']) < self.max_questions)
    
    def force_complete(self) -> Dict[str, any]:
        """Принудительно завершает экзамен досрочно"""
        if self.exam_session['status'] == 'in_progress':
            return self.complete_exam()
        else:
            return {'error': 'Экзамен не активен'}
    
    def get_session_info(self) -> Dict[str, any]:
        """Возвращает информацию о текущей сессии"""
        session_info = {
            'session_id': self.exam_session['session_id'],
            'student_name': self.exam_session['student_name'],
            'subject': self.subject,
            'difficulty': self.difficulty,
            'status': self.exam_session['status'],
            'start_time': self.exam_session['start_time'].isoformat() if self.exam_session['start_time'] else None,
            'end_time': self.exam_session['end_time'].isoformat() if self.exam_session['end_time'] else None,
            'questions_count': len(self.exam_session['questions']),
            'evaluations_count': len(self.exam_session['evaluations']),
            'use_theme_structure': self.use_theme_structure
        }
        
        # Добавляем информацию о тематической структуре, если используется
        if self.use_theme_structure and self.question_agent:
            theme_progress = self.question_agent.get_theme_progress()
            session_info['theme_progress'] = theme_progress
        
        return session_info
    
    def get_theme_structure_info(self) -> Dict[str, any]:
        """Возвращает информацию о тематической структуре экзамена"""
        if not self.use_theme_structure or not self.theme_structure:
            return {'error': 'Тематическая структура не используется'}
        
        return {
            'curriculum_id': self.theme_structure.get('curriculum_id'),
            'total_questions': self.theme_structure.get('total_questions'),
            'questions_distribution': self.theme_structure.get('questions_distribution'),
            'bloom_coverage': self.theme_structure.get('metadata', {}).get('bloom_coverage'),
            'estimated_duration': self.theme_structure.get('metadata', {}).get('estimated_duration'),
            'assessment_framework': self.theme_structure.get('assessment_framework'),
            'question_guidelines': self.theme_structure.get('question_guidelines')
        }
    
    def get_theme_summary_report(self) -> str:
        """Генерирует краткий отчет о тематической структуре экзамена"""
        if not self.use_theme_structure or not self.theme_agent or not self.theme_structure:
            return "Тематическая структура не используется в данном экзамене."
        
        return self.theme_agent.generate_summary_report(self.theme_structure)
    
    def validate_theme_structure(self) -> Dict[str, any]:
        """Валидирует тематическую структуру экзамена"""
        if not self.use_theme_structure or not self.theme_agent or not self.theme_structure:
            return {'error': 'Тематическая структура не используется'}
        
        return self.theme_agent.validate_theme_structure(self.theme_structure)
    
    def export_theme_structure(self, format: str = 'json') -> str:
        """Экспортирует тематическую структуру экзамена"""
        if not self.use_theme_structure or not self.theme_agent or not self.theme_structure:
            return '{"error": "Тематическая структура не используется"}'
        
        if format == 'json':
            return self.theme_agent.export_structure_to_json(self.theme_structure)
        else:
            return str(self.theme_structure)
    
    def get_theme_progress_detailed(self) -> Dict[str, any]:
        """Возвращает детальный прогресс по тематической структуре"""
        if not self.use_theme_structure or not self.question_agent:
            return {'error': 'Тематическая структура не используется'}
        
        progress = self.question_agent.get_theme_progress()
        
        # Добавляем информацию о вопросах по уровням из руководящих принципов
        if self.theme_structure:
            question_guidelines = self.theme_structure.get('question_guidelines', {})
            levels_info = {}
            
            for level, guidelines in question_guidelines.items():
                levels_info[level] = {
                    'total_questions': guidelines.get('question_count', 0),
                    'completed_questions': 0,  # Можно добавить логику подсчета
                    'level_name': guidelines.get('level_name', level),
                    'description': self._get_bloom_level_description(level),
                    'has_guidelines': bool(guidelines.get('guidelines'))
                }
            
            progress['levels_detailed'] = levels_info
        
        return progress
    
    def _get_bloom_level_name(self, level: str) -> str:
        """Возвращает название уровня Блума на русском"""
        names = {
            'remember': 'Запоминание',
            'understand': 'Понимание',
            'apply': 'Применение', 
            'analyze': 'Анализ',
            'evaluate': 'Оценивание',
            'create': 'Создание'
        }
        return names.get(level, level)
    
    def _get_bloom_level_description(self, level: str) -> str:
        """Возвращает описание уровня Блума"""
        descriptions = {
            'remember': 'Извлечение знаний из долговременной памяти',
            'understand': 'Понимание значения материала',
            'apply': 'Использование знаний в новых ситуациях',
            'analyze': 'Разделение на части и понимание связей',
            'evaluate': 'Формирование суждений на основе критериев',
            'create': 'Создание нового продукта или точки зрения'
        }
        return descriptions.get(level, '')