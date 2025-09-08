"""
Агент для генерации вопросов на основе темы и предыдущих ответов студента
"""
from typing import Dict, List, Optional
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from yagpt_llm import YandexGPT
import json
import re


class QuestionAgent:
    """Агент для умной генерации вопросов с учетом контекста"""
    
    def __init__(self, subject: str = "Общие знания", difficulty: str = "средний", topic_context: str = None, theme_structure: dict = None):
        """
        Инициализация агента
        
        Args:
            subject: Предмет экзамена
            difficulty: Уровень сложности (легкий, средний, сложный)
            topic_context: Контекст конкретной темы экзамена
            theme_structure: Тематическая структура от ThemeAgent
        """
        self.llm = YandexGPT()
        self.subject = subject
        self.difficulty = difficulty
        self.topic_context = topic_context or f"Общий экзамен по предмету {subject}"
        self.theme_structure = theme_structure
        self.question_history = []
        self.answer_history = []
        self.current_theme_position = 0  # Позиция в тематической последовательности
        
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Настройка промптов для генерации вопросов"""
        
        # Промпт для первого вопроса
        self.initial_question_prompt = PromptTemplate(
            input_variables=["subject", "difficulty", "topic_context"],
            template="""
Ты эксперт-экзаменатор по предмету "{subject}".

{topic_context}

Создай первый вопрос для экзамена уровня сложности "{difficulty}" строго по указанной теме.

Требования к вопросу:
- Вопрос должен быть базовым для понимания конкретной темы
- Проверяет фундаментальные знания именно по данной теме
- Четкий и однозначный
- Позволяет оценить уровень студента в данной области
- НЕ отклоняйся от заданной темы

Формат ответа:
ВОПРОС: [текст вопроса]
КЛЮЧЕВЫЕ_МОМЕНТЫ: [список ключевых моментов через запятую]

Пример для темы "Циклы в Python":
ВОПРОС: Объясните разницу между циклами for и while в Python
КЛЮЧЕВЫЕ_МОМЕНТЫ: итерация по последовательности, условие выполнения, использование range, бесконечные циклы
"""
        )
        
        # Промпт для последующих вопросов с учетом контекста
        self.contextual_question_prompt = PromptTemplate(
            input_variables=["subject", "difficulty", "question_number", "topic_context", "previous_questions", "previous_answers"],
            template="""
Ты эксперт-экзаменатор по предмету "{subject}".

{topic_context}

Создай вопрос номер {question_number} уровня сложности "{difficulty}" строго по указанной теме.

ПРЕДЫДУЩИЕ ВОПРОСЫ:
{previous_questions}

ОТВЕТЫ СТУДЕНТА:
{previous_answers}

Требования к новому вопросу:
- НЕ повторяй уже заданные вопросы
- Вопрос должен быть строго по заданной теме
- Учитывай характеристики предыдущих ответов для адаптации сложности
- Строй логическую последовательность изучения конкретной темы
- НЕ отклоняйся от основной темы экзамена

Формат ответа:
ВОПРОС: [текст вопроса]
КЛЮЧЕВЫЕ_МОМЕНТЫ: [список ключевых моментов через запятую]
УРОВЕНЬ_ТЕМЫ: [базовый/промежуточный/продвинутый]
ОБОСНОВАНИЕ: [почему этот вопрос подходит для данного студента]

Пример для темы "Циклы в Python":
ВОПРОС: Напишите код для поиска всех четных чисел в списке используя цикл for и list comprehension
КЛЮЧЕВЫЕ_МОМЕНТЫ: цикл for, условие if, модуль %, list comprehension, синтаксис
УРОВЕНЬ_ТЕМЫ: промежуточный
ОБОСНОВАНИЕ: Студент понял базовые циклы, можно переходить к практическому применению
"""
        )
        
        # Промпт для генерации вопросов на основе руководящих принципов ThemeAgent
        self.theme_guided_question_prompt = PromptTemplate(
            input_variables=["subject", "topic_context", "difficulty", "question_requirements", "evaluation_characteristics"],
            template="""
Ты эксперт-экзаменатор по предмету "{subject}".

КОНТЕКСТ ТЕМЫ:
{topic_context}

УРОВЕНЬ СЛОЖНОСТИ: {difficulty}

ТРЕБОВАНИЯ К ВОПРОСУ (от ThemeAgent):
{question_requirements}

ХАРАКТЕРИСТИКИ ПРЕДЫДУЩИХ ОТВЕТОВ (от EvaluationAgent):
{evaluation_characteristics}

ЗАДАЧА:
Создай ОДИН конкретный вопрос, строго следуя требованиям ThemeAgent.
Адаптируй вопрос на основе характеристик предыдущих ответов, но сохраняй заданный уровень Блума.

ВАЖНО: 
- Ты НЕ ВИДИШЬ текстов ответов студента для сохранения приватности
- Ты получаешь только обобщенные характеристики от EvaluationAgent
- Адаптируй вопросы на основе паттернов успеваемости, а не конкретного содержания

ОБЯЗАТЕЛЬНЫЕ ТРЕБОВАНИЯ:
1. Следуй принципам формулирования из требований ThemeAgent
2. Включи все обязательные элементы
3. Используй указанные тематические направления
4. Применяй рекомендованные глаголы и действия
5. Соблюдай указанный уровень сложности
6. Учитывай контекстные требования
7. Избегай того, что указано в разделе "избегать"
8. Адаптируй под характеристики успеваемости студента

ФОРМАТ ОТВЕТА:
ВОПРОС: [конкретный текст вопроса]
КЛЮЧЕВЫЕ_МОМЕНТЫ: [что должно быть в правильном ответе]
УРОВЕНЬ_БЛУМА: [уровень из требований]
ТЕМАТИЧЕСКОЕ_НАПРАВЛЕНИЕ: [какое направление из требований использовано]
КОГНИТИВНЫЙ_ПРОЦЕСС: [какой процесс проверяется]
КРИТЕРИИ_ОЦЕНКИ: [как оценивать этот конкретный ответ]
АДАПТАЦИЯ: [как вопрос адаптирован под характеристики студента]
ПРИВАТНОСТЬ: [подтверждение что текст ответов не использовался]

НЕ отклоняйся от требований ThemeAgent! Создавай только то, что соответствует заданной структуре.
"""
        )
    
    def generate_question(self, question_number: int, evaluation_summaries: List[Dict] = None) -> Dict[str, str]:
        """
        Генерирует вопрос с учетом характеристик от EvaluationAgent (БЕЗ текстов ответов)
        
        Args:
            question_number: Номер вопроса
            evaluation_summaries: Список характеристик оценок от EvaluationAgent (БЕЗ текстов ответов)
            
        Returns:
            Словарь с вопросом и метаданными
        """
        # Если есть тематическая структура от ThemeAgent, используем её
        if self.theme_structure:
            return self._generate_theme_guided_question(question_number, evaluation_summaries)
        
        # Иначе используем обычную логику
        if question_number == 1 or not evaluation_summaries:
            return self._generate_initial_question()
        else:
            # Для обычной логики конвертируем evaluation_summaries в старый формат
            legacy_answers = self._convert_summaries_to_legacy_format(evaluation_summaries)
            return self._generate_contextual_question(question_number, legacy_answers)
    
    def _generate_initial_question(self) -> Dict[str, str]:
        """Генерирует первый вопрос"""
        chain = LLMChain(llm=self.llm, prompt=self.initial_question_prompt)
        
        response = chain.run(
            subject=self.subject,
            difficulty=self.difficulty,
            topic_context=self.topic_context
        )
        
        question_data = self._parse_question_response(response)
        self.question_history.append(question_data)
        
        return question_data
    
    def _generate_contextual_question(self, question_number: int, previous_answers: List[Dict]) -> Dict[str, str]:
        """Генерирует вопрос с учетом контекста"""
        
        # Анализ предыдущих вопросов и ответов
        previous_questions_text = self._format_previous_questions()
        previous_answers_text = self._format_previous_answers(previous_answers)
        
        chain = LLMChain(llm=self.llm, prompt=self.contextual_question_prompt)
        
        response = chain.run(
            subject=self.subject,
            difficulty=self.difficulty,
            question_number=question_number,
            topic_context=self.topic_context,
            previous_questions=previous_questions_text,
            previous_answers=previous_answers_text
        )
        
        question_data = self._parse_question_response(response)
        self.question_history.append(question_data)
        
        return question_data
    
    def _format_previous_questions(self) -> str:
        """Форматирует предыдущие вопросы для промпта"""
        if not self.question_history:
            return "Нет предыдущих вопросов"
        
        formatted = ""
        for i, q in enumerate(self.question_history, 1):
            formatted += f"{i}. {q['question']}\n"
        
        return formatted
    
    def _format_previous_answers(self, previous_answers: List[Dict]) -> str:
        """Форматирует предыдущие ответы для промпта"""
        if not previous_answers:
            return "Нет предыдущих ответов"
        
        formatted = ""
        for i, answer in enumerate(previous_answers, 1):
            formatted += f"{i}. {answer.get('answer', 'Нет ответа')}\n"
            formatted += f"   Оценка: {answer.get('score', 0)}/10\n"
            if 'feedback' in answer:
                formatted += f"   Комментарий: {answer['feedback']}\n"
            formatted += "\n"
        
        return formatted
    
    
    def _parse_question_response(self, response: str) -> Dict[str, str]:
        """Парсит ответ с вопросом"""
        question_match = re.search(r'ВОПРОС:\s*(.+?)(?=\n|$)', response, re.DOTALL)
        key_points_match = re.search(r'КЛЮЧЕВЫЕ_МОМЕНТЫ:\s*(.+?)(?=\n|$)', response)
        level_match = re.search(r'УРОВЕНЬ_ТЕМЫ:\s*(.+?)(?=\n|$)', response)
        reasoning_match = re.search(r'ОБОСНОВАНИЕ:\s*(.+?)(?=\n|$)', response, re.DOTALL)
        
        return {
            'question': question_match.group(1).strip() if question_match else "Вопрос не найден",
            'key_points': key_points_match.group(1).strip() if key_points_match else "",
            'topic_level': level_match.group(1).strip() if level_match else "базовый",
            'reasoning': reasoning_match.group(1).strip() if reasoning_match else "",
            'raw_response': response
        }
    
    def get_question_history(self) -> List[Dict]:
        """Возвращает историю заданных вопросов"""
        return self.question_history.copy()
    
    def _generate_theme_guided_question(self, question_number: int, evaluation_summaries: List[Dict] = None) -> Dict[str, str]:
        """
        Генерирует вопрос согласно требованиям ThemeAgent на основе характеристик от EvaluationAgent
        
        Args:
            question_number: Номер вопроса
            evaluation_summaries: Список характеристик оценок (БЕЗ текстов ответов)
            
        Returns:
            Словарь с вопросом и метаданными
        """
        # Получаем требования для следующего вопроса от ThemeAgent
        question_requirements = self._get_next_question_requirements()
        
        if not question_requirements:
            # Если требования закончились, генерируем обычный вопрос
            return self._generate_contextual_question(question_number, [])
        
        # Генерируем вопрос на основе требований ThemeAgent и характеристик оценок
        theme_question = self._create_question_from_requirements(question_requirements, evaluation_summaries)
        
        self.question_history.append(theme_question)
        self.current_theme_position += 1
        
        return theme_question
    
    def _get_next_question_requirements(self) -> Optional[Dict]:
        """Получает требования для следующего вопроса от ThemeAgent"""
        if not self.theme_structure:
            return None
        
        # Используем метод ThemeAgent для получения требований
        try:
            from theme_agent import ThemeAgent
            theme_agent = ThemeAgent()
            requirements = theme_agent.get_next_bloom_level_requirements(
                self.theme_structure, 
                self.current_theme_position
            )
            return requirements
        except:
            return None
    
    def _create_question_from_requirements(self, requirements: Dict, evaluation_summaries: List[Dict] = None) -> Dict[str, str]:
        """
        Создает вопрос на основе требований ThemeAgent и характеристик от EvaluationAgent
        
        Args:
            requirements: Требования от ThemeAgent
            evaluation_summaries: Обобщенные характеристики от EvaluationAgent (БЕЗ текстов ответов)
            
        Returns:
            Сгенерированный вопрос
        """
        # Подготавливаем характеристики оценок (без текстов ответов)
        evaluation_characteristics = self._format_evaluation_characteristics(evaluation_summaries or [])
        
        # Форматируем требования для промпта
        requirements_text = self._format_requirements_for_prompt(requirements)
        
        # Генерируем вопрос через LLM
        chain = LLMChain(llm=self.llm, prompt=self.theme_guided_question_prompt)
        
        response = chain.run(
            subject=self.subject,
            topic_context=self.topic_context,
            difficulty=self.difficulty,
            question_requirements=requirements_text,
            evaluation_characteristics=evaluation_characteristics
        )
        
        # Парсим и структурируем ответ
        return self._parse_theme_guided_question(response, requirements)
    
    def _format_requirements_for_prompt(self, requirements: Dict) -> str:
        """Форматирует требования ThemeAgent для промпта"""
        if not requirements or 'error' in requirements:
            return "Требования не доступны"
        
        text = f"УРОВЕНЬ БЛУМА: {requirements.get('level_name', 'Не указан')}\n"
        text += f"КОЛИЧЕСТВО ВОПРОСОВ ЭТОГО ТИПА: {requirements.get('question_count', 1)}\n\n"
        
        if requirements.get('formulation_principles'):
            text += f"ПРИНЦИПЫ ФОРМУЛИРОВАНИЯ:\n{requirements['formulation_principles']}\n\n"
        
        if requirements.get('mandatory_elements'):
            text += f"ОБЯЗАТЕЛЬНЫЕ ЭЛЕМЕНТЫ:\n{requirements['mandatory_elements']}\n\n"
        
        if requirements.get('thematic_directions'):
            text += f"ТЕМАТИЧЕСКИЕ НАПРАВЛЕНИЯ:\n{requirements['thematic_directions']}\n\n"
        
        if requirements.get('verbs_and_actions'):
            text += f"РЕКОМЕНДУЕМЫЕ ГЛАГОЛЫ И ДЕЙСТВИЯ:\n{requirements['verbs_and_actions']}\n\n"
        
        if requirements.get('complexity_level'):
            text += f"ТРЕБОВАНИЯ К СЛОЖНОСТИ:\n{requirements['complexity_level']}\n\n"
        
        if requirements.get('contextual_requirements'):
            text += f"КОНТЕКСТНЫЕ ТРЕБОВАНИЯ:\n{requirements['contextual_requirements']}\n\n"
        
        if requirements.get('quality_criteria'):
            text += f"КРИТЕРИИ КАЧЕСТВА:\n{requirements['quality_criteria']}\n\n"
        
        if requirements.get('avoid'):
            text += f"ИЗБЕГАТЬ:\n{requirements['avoid']}\n\n"
        
        if requirements.get('student_adaptation'):
            text += f"АДАПТАЦИЯ ПОД СТУДЕНТА:\n{requirements['student_adaptation']}\n\n"
        
        if requirements.get('format_requirements'):
            text += f"ТРЕБОВАНИЯ К ФОРМАТУ:\n{requirements['format_requirements']}\n\n"
        
        return text
    
    def _format_evaluation_characteristics(self, evaluation_summaries: List[Dict]) -> str:
        """
        Форматирует характеристики оценок БЕЗ текстов ответов студента
        
        Args:
            evaluation_summaries: Список характеристик от EvaluationAgent
            
        Returns:
            Форматированный текст характеристик
        """
        if not evaluation_summaries:
            return "Предыдущих оценок нет - это первый вопрос."
        
        characteristics = []
        
        for i, summary in enumerate(evaluation_summaries, 1):
            char_text = f"ОТВЕТ {i}:\n"
            
            # Оценки по критериям (БЕЗ содержания ответа)
            if 'criteria_scores' in summary:
                scores = summary['criteria_scores']
                char_text += f"  • Правильность: {scores.get('correctness', 0)}/10\n"
                char_text += f"  • Полнота: {scores.get('completeness', 0)}/10\n"
                char_text += f"  • Понимание: {scores.get('understanding', 0)}/10\n"
                char_text += f"  • Структурированность: {scores.get('structure', 0)}/10\n"
            
            # Общий балл
            char_text += f"  • Общий балл: {summary.get('total_score', 0)}/10\n"
            
            # Сильные стороны (обобщенно)
            if 'strengths' in summary:
                char_text += f"  • Сильные стороны: {summary['strengths'][:100]}...\n"
            
            # Области для улучшения (обобщенно)
            if 'weaknesses' in summary:
                char_text += f"  • Слабые стороны: {summary['weaknesses'][:100]}...\n"
            
            # Уровень Блума
            if 'bloom_level' in summary:
                char_text += f"  • Уровень Блума: {summary['bloom_level']}\n"
            
            characteristics.append(char_text)
        
        return "\n".join(characteristics)
    
    
    def _convert_summaries_to_legacy_format(self, evaluation_summaries: List[Dict]) -> List[Dict]:
        """
        Конвертирует evaluation_summaries в legacy формат для совместимости
        НО БЕЗ текстов ответов для сохранения приватности
        
        Args:
            evaluation_summaries: Характеристики от EvaluationAgent
            
        Returns:
            Legacy формат (без текстов ответов)
        """
        legacy_format = []
        
        for summary in evaluation_summaries:
            legacy_item = {
                'score': summary.get('total_score', 0),
                'bloom_level': summary.get('bloom_level', 'unknown'),
                'criteria_scores': summary.get('criteria_scores', {}),
                'strengths': summary.get('strengths', ''),
                'weaknesses': summary.get('weaknesses', ''),
                # НЕ ВКЛЮЧАЕМ: 'answer_text' для сохранения приватности
                'evaluation_metadata': {
                    'timestamp': summary.get('timestamp'),
                    'question_type': summary.get('question_type')
                }
            }
            legacy_format.append(legacy_item)
        
        return legacy_format
    
    def _parse_theme_guided_question(self, response: str, requirements: Dict) -> Dict[str, str]:
        """Парсит вопрос, сгенерированный на основе требований ThemeAgent"""
        question_match = re.search(r'ВОПРОС:\s*(.+?)(?=\n|$)', response, re.DOTALL)
        key_points_match = re.search(r'КЛЮЧЕВЫЕ_МОМЕНТЫ:\s*(.+?)(?=\n|$)', response)
        bloom_level_match = re.search(r'УРОВЕНЬ_БЛУМА:\s*(.+?)(?=\n|$)', response)
        direction_match = re.search(r'ТЕМАТИЧЕСКОЕ_НАПРАВЛЕНИЕ:\s*(.+?)(?=\n|$)', response)
        process_match = re.search(r'КОГНИТИВНЫЙ_ПРОЦЕСС:\s*(.+?)(?=\n|$)', response)
        criteria_match = re.search(r'КРИТЕРИИ_ОЦЕНКИ:\s*(.+?)(?=\n|$)', response, re.DOTALL)
        adaptation_match = re.search(r'АДАПТАЦИЯ:\s*(.+?)(?=\n|$)', response, re.DOTALL)
        
        return {
            'question': question_match.group(1).strip() if question_match else "Вопрос не найден",
            'key_points': key_points_match.group(1).strip() if key_points_match else "",
            'topic_level': self._map_bloom_to_topic_level(requirements.get('bloom_level', 'remember')),
            'bloom_level': requirements.get('bloom_level', 'remember'),
            'bloom_level_name': requirements.get('level_name', 'Не указан'),
            'thematic_direction': direction_match.group(1).strip() if direction_match else "",
            'cognitive_process': process_match.group(1).strip() if process_match else "",
            'evaluation_criteria': criteria_match.group(1).strip() if criteria_match else "",
            'adaptation_notes': adaptation_match.group(1).strip() if adaptation_match else "",
            'theme_requirements': requirements,
            'raw_response': response
        }
    
    def _map_bloom_to_topic_level(self, bloom_level: str) -> str:
        """Преобразует уровень Блума в уровень темы"""
        mapping = {
            'remember': 'базовый',
            'understand': 'базовый',
            'apply': 'промежуточный',
            'analyze': 'промежуточный',
            'evaluate': 'продвинутый',
            'create': 'продвинутый'
        }
        return mapping.get(bloom_level, 'базовый')
    
    def get_theme_progress(self) -> Dict[str, any]:
        """Возвращает прогресс по тематической структуре"""
        if not self.theme_structure:
            return {'error': 'Тематическая структура не загружена'}
        
        distribution = self.theme_structure.get('questions_distribution', {})
        total_questions = sum(distribution.values())
        
        return {
            'current_position': self.current_theme_position,
            'total_questions': total_questions,
            'progress_percentage': (self.current_theme_position / total_questions * 100) if total_questions > 0 else 0,
            'current_bloom_level': self._get_current_theme_level(),
            'completed_levels': self._get_completed_theme_levels(),
            'remaining_levels': self._get_remaining_theme_levels(),
            'theme_structure_id': self.theme_structure.get('curriculum_id', 'unknown')
        }
    
    def _get_current_theme_level(self) -> str:
        """Определяет текущий уровень в тематической структуре"""
        requirements = self._get_next_question_requirements()
        if requirements and 'bloom_level' in requirements:
            return requirements['bloom_level']
        return 'completed'
    
    def _get_completed_theme_levels(self) -> List[str]:
        """Возвращает завершенные уровни в тематической структуре"""
        if not self.theme_structure:
            return []
        
        distribution = self.theme_structure.get('questions_distribution', {})
        bloom_sequence = self.theme_structure.get('bloom_sequence', ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create'])
        
        completed = []
        current_pos = 0
        
        for level in bloom_sequence:
            if level in distribution:
                level_questions_count = distribution[level]
                if self.current_theme_position >= current_pos + level_questions_count:
                    completed.append(level)
                current_pos += level_questions_count
        
        return completed
    
    def _get_remaining_theme_levels(self) -> List[str]:
        """Возвращает оставшиеся уровни в тематической структуре"""
        if not self.theme_structure:
            return []
        
        all_levels = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
        completed = self._get_completed_theme_levels()
        current_level = self._get_current_theme_level()
        
        remaining = []
        for level in all_levels:
            if level not in completed and level != current_level and level != 'completed':
                remaining.append(level)
        
        return remaining
    
    def reset_history(self):
        """Сбрасывает историю вопросов"""
        self.question_history = []
        self.answer_history = []
        self.current_theme_position = 0