"""
Агент для оценки ответов студентов на LangGraph
"""
from typing import Dict, List, Optional, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from base import LangGraphAgentBase, EvaluationState
from yagpt_llm import create_yandex_llm
import json
import re
from datetime import datetime


class EvaluationAgentLangGraph(LangGraphAgentBase):
    """Агент для объективной изолированной оценки ответов на LangGraph"""
    
    def __init__(self, subject: str = "Общие знания", topic_context: str = None):
        super().__init__(subject, topic_context)
        self.llm = create_yandex_llm()
        self.evaluation_history = []
        
        # Создаем граф состояний
        self.graph = self._create_evaluation_graph()
        self.app = self.graph.compile()
        
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Настройка промптов для оценки"""
        
        # Основной промпт для оценки ответа
        self.evaluation_prompt = PromptTemplate(
            input_variables=["subject", "topic_context", "question", "student_answer", "key_points", "topic_level"],
            template="""
Ты строгий и объективный экзаменатор по предмету "{subject}".

{topic_context}

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
        
    
    def _create_evaluation_graph(self) -> StateGraph:
        """Создает граф состояний для оценки ответов"""
        graph = StateGraph(EvaluationState)
        
        # Добавляем узлы
        graph.add_node("validate_input", self._validate_input_node)
        graph.add_node("handle_empty_answer", self._handle_empty_answer_node)
        graph.add_node("detailed_evaluation", self._detailed_evaluation_node)
        graph.add_node("parse_evaluation", self._parse_evaluation_node)
        graph.add_node("create_summary", self._create_summary_node)
        graph.add_node("save_to_history", self._save_to_history_node)
        
        # Определяем точку входа
        graph.set_entry_point("validate_input")
        
        # Добавляем условные ребра
        graph.add_conditional_edges(
            "validate_input",
            self._decide_evaluation_path,
            {
                "empty_answer": "handle_empty_answer",
                "detailed": "detailed_evaluation"
            }
        )
        
        # Ребра к парсингу
        graph.add_edge("detailed_evaluation", "parse_evaluation")
        graph.add_edge("handle_empty_answer", "create_summary")  # Пустые ответы идут сразу к созданию summary
        
        # Ребра к созданию summary
        graph.add_edge("parse_evaluation", "create_summary")
        
        # Ребра к сохранению в историю
        graph.add_edge("create_summary", "save_to_history")
        
        # Завершение
        graph.add_edge("save_to_history", END)
        
        return graph
    
    def _validate_input_node(self, state: EvaluationState) -> EvaluationState:
        """Валидирует входные данные"""
        try:
            # Проверяем обязательные поля
            if not state.get("question"):
                state["error"] = "Отсутствует вопрос"
                return state
            
            if not state.get("student_answer") or state["student_answer"].strip() == "":
                state["evaluation_type"] = "empty"
                return state
            
            # Всегда используем детальную оценку
            state["evaluation_type"] = "detailed"
            
            self.log_operation("validate_input", state, None)
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка валидации входных данных: {str(e)}"
            self.log_operation("validate_input", state, None, str(e))
            return state
    
    def _decide_evaluation_path(self, state: EvaluationState) -> str:
        """Решает, какой путь использовать для оценки"""
        if state.get("error"):
            return "empty_answer"  # Fallback для ошибок
        
        evaluation_type = state.get("evaluation_type", "detailed")
        
        if evaluation_type == "empty":
            return "empty_answer"
        else:
            return "detailed"
    
    def _handle_empty_answer_node(self, state: EvaluationState) -> EvaluationState:
        """Обрабатывает случай пустого ответа"""
        try:
            empty_result = {
                'type': 'empty',
                'total_score': 0,
                'comment': "Ответ не предоставлен",
                'feedback': "Студент не дал ответ на вопрос",
                'raw_response': "EMPTY_ANSWER",
                'timestamp': datetime.now()
            }
            
            state["evaluation_result"] = empty_result
            self.log_operation("handle_empty_answer", state, empty_result)
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка обработки пустого ответа: {str(e)}"
            self.log_operation("handle_empty_answer", state, None, str(e))
            return state
    
    def _detailed_evaluation_node(self, state: EvaluationState) -> EvaluationState:
        """Выполняет детальную оценку ответа"""
        try:
            chain = self.evaluation_prompt | self.llm | StrOutputParser()
            
            response = chain.invoke({
                "subject": self.subject,
                "topic_context": self.topic_context,
                "question": state["question"],
                "student_answer": state["student_answer"],
                "key_points": state.get("key_points", ""),
                "topic_level": state.get("topic_level", "базовый")
            })
            
            print(response["topic_context"])
            state["raw_evaluation"] = response
            state["evaluation_type"] = "detailed"
            self.log_operation("detailed_evaluation", state, response)
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка детальной оценки: {str(e)}"
            self.log_operation("detailed_evaluation", state, None, str(e))
            return state
    
    
    def _parse_evaluation_node(self, state: EvaluationState) -> EvaluationState:
        """Парсит результат оценки"""
        try:
            if state.get("error") or not state.get("raw_evaluation"):
                return state
            
            response = state["raw_evaluation"]
            evaluation_type = state.get("evaluation_type", "detailed")
            
            # Всегда используем детальный парсинг
            evaluation_result = self._parse_detailed_evaluation(response)
            
            # Добавляем метаданные
            evaluation_result.update({
                'timestamp': datetime.now(),
                'question_metadata': state.get("question_metadata", {})
            })
            
            state["evaluation_result"] = evaluation_result
            self.log_operation("parse_evaluation", response, evaluation_result)
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка парсинга оценки: {str(e)}"
            self.log_operation("parse_evaluation", state, None, str(e))
            return state
    
    def _create_summary_node(self, state: EvaluationState) -> EvaluationState:
        """Создает сводку оценки БЕЗ текста ответа для передачи QuestionAgent"""
        try:
            evaluation_result = state.get("evaluation_result")
            if not evaluation_result:
                state["error"] = "Нет результата оценки для создания сводки"
                return state
            
            question_metadata = state.get("question_metadata", {})
            
            summary = {
                # Основные оценки
                'total_score': evaluation_result.get('total_score', 0),
                'criteria_scores': evaluation_result.get('criteria_scores', {}),
                
                # Характеристики качества ответа (обобщенные)
                'strengths': evaluation_result.get('strengths', ''),
                'weaknesses': evaluation_result.get('weaknesses', ''),
                
                # Метаданные вопроса
                'bloom_level': question_metadata.get('bloom_level', 'unknown'),
                'question_type': question_metadata.get('question_type', 'unknown'),
                'topic_level': question_metadata.get('topic_level', 'unknown'),
                
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
                }
                
                # ВАЖНО: НЕ ВКЛЮЧАЕМ текст ответа студента для приватности
                # 'student_answer': evaluation_result.get('student_answer')  # <-- ИСКЛЮЧЕНО
            }
            
            state["evaluation_summary"] = summary
            self.log_operation("create_summary", evaluation_result, summary)
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка создания сводки: {str(e)}"
            self.log_operation("create_summary", state, None, str(e))
            return state
    
    def _save_to_history_node(self, state: EvaluationState) -> EvaluationState:
        """Сохраняет результат в историю"""
        try:
            evaluation_result = state.get("evaluation_result")
            if evaluation_result and not state.get("error"):
                # Сохраняем в историю с полной информацией (включая текст ответа)
                history_entry = {
                    'question': state["question"],
                    'answer': state["student_answer"],  # Сохраняем для DiagnosticAgent
                    'evaluation': evaluation_result,
                    'timestamp': evaluation_result.get('timestamp'),
                    'question_metadata': state.get("question_metadata", {})
                }
                
                self.evaluation_history.append(history_entry)
                
                self.log_operation("save_to_history", evaluation_result, "Saved successfully")
            
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка сохранения в историю: {str(e)}"
            self.log_operation("save_to_history", state, None, str(e))
            return state
    
    def _parse_detailed_evaluation(self, response: str) -> Dict[str, Any]:
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
    
    def evaluate_answer(self, question: str, student_answer: str, key_points: str, 
                       topic_level: str = "базовый", detailed: bool = True) -> Dict[str, Any]:
        """
        Оценивает ответ студента изолированно с использованием LangGraph
        
        Args:
            question: Текст вопроса
            student_answer: Ответ студента
            key_points: Ключевые моменты для ответа
            topic_level: Уровень сложности темы
            detailed: Использовать детальную оценку или быструю
            
        Returns:
            Результат оценки
        """
        try:
            # Создаем начальное состояние
            initial_state = EvaluationState(
                question=question,
                student_answer=student_answer,
                key_points=key_points,
                topic_level=topic_level,
                question_metadata={'detailed': detailed},
                evaluation_result=None,
                evaluation_summary=None,
                error=None
            )
            
            # Запускаем граф
            result = self.app.invoke(initial_state)
            
            # Проверяем результат
            if result.get("error"):
                return {
                    "error": result["error"],
                    "type": "error",
                    "total_score": 0,
                    "timestamp": datetime.now()
                }
            
            return result.get("evaluation_result", {})
            
        except Exception as e:
            error_msg = f"Критическая ошибка в evaluate_answer: {str(e)}"
            self.log_operation("evaluate_answer", {
                "question": question[:100],
                "answer": student_answer[:100]
            }, None, error_msg)
            
            return {
                "error": error_msg,
                "type": "error", 
                "total_score": 0,
                "timestamp": datetime.now()
            }
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
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
    
    def get_evaluation_summaries_for_question_agent(self) -> List[Dict]:
        """
        Возвращает список сводок оценок БЕЗ текстов ответов для QuestionAgent
        
        Returns:
            Список характеристик оценок без приватной информации
        """
        summaries = []
        
        for evaluation in self.evaluation_history:
            # Получаем метаданные вопроса из истории оценок
            question_metadata = evaluation.get('question_metadata', {})
            
            # Создаем summary без текста ответа
            summary = self.create_evaluation_summary(evaluation['evaluation'], question_metadata)
            summaries.append(summary)
        
        return summaries
    
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
            }
            
            # ВАЖНО: НЕ ВКЛЮЧАЕМ текст ответа студента для приватности
            # 'student_answer': evaluation_result.get('student_answer')  # <-- ИСКЛЮЧЕНО
        }
        
        return summary
    
    def reset_history(self):
        """Сбрасывает историю оценок"""
        self.evaluation_history = []
        super().reset_history()


# Функция для создания EvaluationAgent на LangGraph
def create_evaluation_agent(
    subject: str = "Общие знания",
    topic_context: str = None
) -> EvaluationAgentLangGraph:
    """Создает экземпляр EvaluationAgent на LangGraph"""
    return EvaluationAgentLangGraph(
        subject=subject,
        topic_context=topic_context
    )

# Псевдоним для обратной совместимости
def create_evaluation_agent_langgraph(
    subject: str = "Общие знания",
    topic_context: str = None
) -> EvaluationAgentLangGraph:
    """Создает экземпляр EvaluationAgent на LangGraph (псевдоним для обратной совместимости)"""
    return create_evaluation_agent(subject, topic_context)
