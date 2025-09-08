"""
Агент для изолированной оценки ответов студентов
"""
from typing import Dict, List, Optional
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from yagpt_llm import YandexGPT
import json
import re


class EvaluationAgent:
    """Агент для объективной изолированной оценки ответов"""
    
    def __init__(self, subject: str = "Общие знания", topic_context: str = None):
        """
        Инициализация агента
        
        Args:
            subject: Предмет экзамена
            topic_context: Контекст конкретной темы экзамена
        """
        self.llm = YandexGPT()
        self.subject = subject
        self.topic_context = topic_context or f"Общий экзамен по предмету {subject}"
        self.evaluation_history = []
        
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Настройка промптов для оценки"""
        
        # Основной промпт для оценки ответа
        self.evaluation_prompt = PromptTemplate(
            input_variables=["subject", "topic_context", "question", "student_answer", "key_points", "topic_level"],
            template="""
Ты строгий и объективный экзаменатор по предмету "{subject}".

{topic_context}

Оцени ТОЛЬКО этот конкретный ответ студента, не учитывая другие ответы или общий контекст экзамена.
Ответ должен соответствовать указанной теме экзамена.

ВОПРОС: {question}

ОТВЕТ СТУДЕНТА: {student_answer}

КРИТЕРИИ ОЦЕНКИ (каждый от 0 до 10 баллов):

1. ПРАВИЛЬНОСТЬ ФАКТОВ (0-10):
   - 9-10: Все факты верны, нет ошибок
   - 7-8: Преимущественно верно, минимальные неточности
   - 5-6: Частично верно, есть ошибки
   - 3-4: Много ошибок, но есть правильные элементы
   - 0-2: Большинство фактов неверны

2. ПОЛНОТА ОТВЕТА (0-10):
   - 9-10: Покрыты все ключевые моменты
   - 7-8: Покрыто большинство ключевых моментов
   - 5-6: Покрыта половина ключевых моментов
   - 3-4: Покрыта малая часть ключевых моментов
   - 0-2: Ключевые моменты почти не затронуты

3. ПОНИМАНИЕ КОНЦЕПЦИЙ (0-10):
   - 9-10: Глубокое понимание, может объяснить принципы
   - 7-8: Хорошее понимание основных концепций
   - 5-6: Поверхностное понимание
   - 3-4: Слабое понимание, путается в концепциях
   - 0-2: Не понимает основных концепций

ОСОБЫЕ ТРЕБОВАНИЯ:
- Будь объективен и справедлив
- Оценивай только содержание ответа
- НЕ учитывай грамматические ошибки, если они не влияют на смысл
- Учитывай уровень сложности темы при оценке

Формат ответа:
ПРАВИЛЬНОСТЬ: [балл]/10 - [краткое обоснование]
ПОЛНОТА: [балл]/10 - [краткое обоснование]
ПОНИМАНИЕ: [балл]/10 - [краткое обоснование]
ИТОГОВАЯ_ОЦЕНКА: [средний балл]/10
ДЕТАЛЬНАЯ_ОБРАТНАЯ_СВЯЗЬ: [подробный анализ ответа]
СИЛЬНЫЕ_СТОРОНЫ: [что хорошо в ответе]
СЛАБЫЕ_СТОРОНЫ: [что плохо в ответе, конкретные ошибки]

Пример:
ПРАВИЛЬНОСТЬ: 8/10 - Основные факты верны, но есть неточность в определении
ПОЛНОТА: 6/10 - Рассмотрены не все ключевые аспекты
ПОНИМАНИЕ: 7/10 - Показано хорошее понимание основных принципов
ИТОГОВАЯ_ОЦЕНКА: 7.0/10
ДЕТАЛЬНАЯ_ОБРАТНАЯ_СВЯЗЬ: Студент демонстрирует хорошее понимание...
СИЛЬНЫЕ_СТОРОНЫ: Правильное понимание основных концепций
СЛАБЫЕ_СТОРОНЫ: Неполное раскрытие темы, неточность в определении цикла for
"""
        )
        
        # Промпт для быстрой оценки (упрощенный)
        self.quick_evaluation_prompt = PromptTemplate(
            input_variables=["question", "student_answer", "key_points"],
            template="""
Быстро и объективно оцени ответ студента.

ВОПРОС: {question}
ОТВЕТ: {student_answer}
КЛЮЧЕВЫЕ МОМЕНТЫ: {key_points}

Оцени по шкале 0-10 и дай краткий комментарий.

Формат:
ОЦЕНКА: [балл]/10
КОММЕНТАРИЙ: [краткое объяснение оценки]
СОВЕТ: [один конкретный совет]
"""
        )
    
    def evaluate_answer(self, question: str, student_answer: str, key_points: str, 
                       topic_level: str = "базовый", detailed: bool = True) -> Dict[str, any]:
        """
        Оценивает ответ студента изолированно
        
        Args:
            question: Текст вопроса
            student_answer: Ответ студента
            key_points: Ключевые моменты для ответа
            topic_level: Уровень сложности темы
            detailed: Использовать детальную оценку или быструю
            
        Returns:
            Результат оценки
        """
        if not student_answer or student_answer.strip() == "":
            return self._handle_empty_answer()
        
        if detailed:
            return self._detailed_evaluation(question, student_answer, key_points, topic_level)
        else:
            return self._quick_evaluation(question, student_answer, key_points)
    
    def _detailed_evaluation(self, question: str, student_answer: str, 
                           key_points: str, topic_level: str) -> Dict[str, any]:
        """Выполняет детальную оценку ответа"""
        chain = LLMChain(llm=self.llm, prompt=self.evaluation_prompt)
        
        response = chain.run(
            subject=self.subject,
            topic_context=self.topic_context,
            question=question,
            student_answer=student_answer,
            key_points=key_points,
            topic_level=topic_level
        )
        
        # Парсинг детальной оценки
        evaluation = self._parse_detailed_evaluation(response)
        
        # Сохранение в историю
        self.evaluation_history.append({
            'question': question,
            'answer': student_answer,
            'evaluation': evaluation,
            'timestamp': None  # Можно добавить datetime
        })
        
        return evaluation
    
    def _quick_evaluation(self, question: str, student_answer: str, key_points: str) -> Dict[str, any]:
        """Выполняет быструю оценку ответа"""
        chain = LLMChain(llm=self.llm, prompt=self.quick_evaluation_prompt)
        
        response = chain.run(
            question=question,
            student_answer=student_answer,
            key_points=key_points
        )
        
        # Парсинг быстрой оценки
        evaluation = self._parse_quick_evaluation(response)
        
        return evaluation
    
    def _parse_detailed_evaluation(self, response: str) -> Dict[str, any]:
        """Парсит детальную оценку"""
        # Извлечение оценок по критериям
        correctness_match = re.search(r'ПРАВИЛЬНОСТЬ:\s*(\d+)/10\s*-\s*(.+?)(?=\n|$)', response)
        completeness_match = re.search(r'ПОЛНОТА:\s*(\d+)/10\s*-\s*(.+?)(?=\n|$)', response)
        understanding_match = re.search(r'ПОНИМАНИЕ:\s*(\d+)/10\s*-\s*(.+?)(?=\n|$)', response)
        
        total_score_match = re.search(r'ИТОГОВАЯ_ОЦЕНКА:\s*([\d.]+)/10', response)
        
        # Извлечение текстовых блоков
        feedback_match = re.search(r'ДЕТАЛЬНАЯ_ОБРАТНАЯ_СВЯЗЬ:\s*(.+?)(?=\nСИЛЬНЫЕ_СТОРОНЫ:|$)', response, re.DOTALL)
        strengths_match = re.search(r'СИЛЬНЫЕ_СТОРОНЫ:\s*(.+?)(?=\nСЛАБЫЕ_СТОРОНЫ:|$)', response, re.DOTALL)
        weaknesses_match = re.search(r'СЛАБЫЕ_СТОРОНЫ:\s*(.+?)(?=\n|$)', response, re.DOTALL)
        
        # Вычисление общего балла (только из 3 критериев)
        scores = []
        if correctness_match:
            scores.append(int(correctness_match.group(1)))
        if completeness_match:
            scores.append(int(completeness_match.group(1)))
        if understanding_match:
            scores.append(int(understanding_match.group(1)))
        
        calculated_score = sum(scores) / len(scores) if scores else 0
        final_score = float(total_score_match.group(1)) if total_score_match else calculated_score
        
        return {
            'type': 'detailed',
            'total_score': round(final_score, 1),
            'criteria_scores': {
                'correctness': int(correctness_match.group(1)) if correctness_match else 0,
                'completeness': int(completeness_match.group(1)) if completeness_match else 0,
                'understanding': int(understanding_match.group(1)) if understanding_match else 0
            },
            'criteria_feedback': {
                'correctness': correctness_match.group(2).strip() if correctness_match else "",
                'completeness': completeness_match.group(2).strip() if completeness_match else "",
                'understanding': understanding_match.group(2).strip() if understanding_match else ""
            },
            'detailed_feedback': feedback_match.group(1).strip() if feedback_match else "",
            'strengths': strengths_match.group(1).strip() if strengths_match else "",
            'weaknesses': weaknesses_match.group(1).strip() if weaknesses_match else "",
            'raw_response': response
        }
    
    def _parse_quick_evaluation(self, response: str) -> Dict[str, any]:
        """Парсит быструю оценку"""
        score_match = re.search(r'ОЦЕНКА:\s*(\d+)/10', response)
        comment_match = re.search(r'КОММЕНТАРИЙ:\s*(.+?)(?=\nСОВЕТ:|$)', response, re.DOTALL)
        advice_match = re.search(r'СОВЕТ:\s*(.+?)(?=\n|$)', response, re.DOTALL)
        
        return {
            'type': 'quick',
            'total_score': int(score_match.group(1)) if score_match else 0,
            'comment': comment_match.group(1).strip() if comment_match else "",
            'advice': advice_match.group(1).strip() if advice_match else "",
            'raw_response': response
        }
    
    def _handle_empty_answer(self) -> Dict[str, any]:
        """Обрабатывает случай пустого ответа"""
        return {
            'type': 'empty',
            'total_score': 0,
            'comment': "Ответ не предоставлен",
            'feedback': "Студент не дал ответ на вопрос",
            'raw_response': "EMPTY_ANSWER"
        }
    
    def get_evaluation_statistics(self) -> Dict[str, any]:
        """Возвращает статистику по оценкам"""
        if not self.evaluation_history:
            return {'message': 'Нет данных для анализа'}
        
        scores = [eval_data['evaluation']['total_score'] for eval_data in self.evaluation_history 
                 if 'total_score' in eval_data['evaluation']]
        
        if not scores:
            return {'message': 'Нет валидных оценок'}
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'average_score': sum(scores) / len(scores),
            'highest_score': max(scores),
            'lowest_score': min(scores),
            'score_distribution': {
                'excellent': len([s for s in scores if s >= 9]),
                'good': len([s for s in scores if 7 <= s < 9]),
                'satisfactory': len([s for s in scores if 5 <= s < 7]),
                'poor': len([s for s in scores if s < 5])
            }
        }
    
    def get_evaluation_history(self) -> List[Dict]:
        """Возвращает историю оценок"""
        return self.evaluation_history.copy()
    
    def create_evaluation_summary(self, evaluation_result: Dict, question_data: Dict = None) -> Dict:
        """
        Создает сводку оценки БЕЗ текста ответа для передачи QuestionAgent
        
        Args:
            evaluation_result: Результат оценки от evaluate_answer
            question_data: Дополнительные данные о вопросе
            
        Returns:
            Сводка характеристик БЕЗ текста ответа студента
        """
        summary = {
            # Основные оценки
            'total_score': evaluation_result.get('total_score', 0),
            'criteria_scores': evaluation_result.get('criteria_scores', {}),
            
            # Характеристики качества ответа (обобщенные)
            'strengths': evaluation_result.get('strengths', ''),
            'weaknesses': evaluation_result.get('weaknesses', ''),
            
            # Метаданные вопроса
            'bloom_level': question_data.get('bloom_level') if question_data else 'unknown',
            'question_type': question_data.get('question_type') if question_data else 'unknown',
            'topic_level': question_data.get('topic_level') if question_data else 'unknown',
            
            # Временные метки
            'timestamp': evaluation_result.get('timestamp'),
            'evaluation_type': evaluation_result.get('type', 'detailed'),
            
            # Индикаторы успеваемости
            'performance_indicators': {
                'correctness_level': self._categorize_score(evaluation_result.get('criteria_scores', {}).get('correctness', 0)),
                'completeness_level': self._categorize_score(evaluation_result.get('criteria_scores', {}).get('completeness', 0)),
                'understanding_level': self._categorize_score(evaluation_result.get('criteria_scores', {}).get('understanding', 0)),
                'structure_level': self._categorize_score(evaluation_result.get('criteria_scores', {}).get('structure', 0)),
                'overall_level': self._categorize_score(evaluation_result.get('total_score', 0))
            },
            
            # ВАЖНО: НЕ ВКЛЮЧАЕМ текст ответа студента для приватности
            # 'student_answer': evaluation_result.get('student_answer')  # <-- ИСКЛЮЧЕНО
        }
        
        return summary
    
    def _categorize_score(self, score: float) -> str:
        """Категоризирует числовую оценку в текстовое описание"""
        if score >= 9:
            return "отлично"
        elif score >= 7:
            return "хорошо"
        elif score >= 5:
            return "удовлетворительно"
        elif score >= 3:
            return "слабо"
        else:
            return "неудовлетворительно"
    
    def get_evaluation_summaries_for_question_agent(self) -> List[Dict]:
        """
        Возвращает список сводок оценок БЕЗ текстов ответов для QuestionAgent
        
        Returns:
            Список характеристик оценок без приватной информации
        """
        summaries = []
        
        for evaluation in self.evaluation_history:
            # Получаем метаданные вопроса из истории оценок
            question_data = evaluation.get('question_metadata', {})
            
            # Создаем summary без текста ответа
            summary = self.create_evaluation_summary(evaluation, question_data)
            summaries.append(summary)
        
        return summaries
    
    def reset_history(self):
        """Сбрасывает историю оценок"""
        self.evaluation_history = []