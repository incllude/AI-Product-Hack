"""
Агент для генерации вопросов на LangGraph
"""
from typing import Dict, List, Optional, Any, TypedDict
from langgraph.graph import StateGraph, END
# ToolExecutor больше не используется в новых версиях LangGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from base import LangGraphAgentBase, QuestionState, ExamState
from yagpt_llm import create_yandex_llm, YandexGPT
import json
import re
from datetime import datetime


class QuestionAgentLangGraph(LangGraphAgentBase):
    """Агент для генерации вопросов с использованием LangGraph"""
    
    def __init__(self, subject: str = "Общие знания", difficulty: str = "средний", 
                 topic_context: str = None, theme_structure: dict = None):
        print(f"🔍 [QuestionAgent] Инициализация агента для предмета: {subject}")
        super().__init__(subject, topic_context)
        self.difficulty = difficulty
        self.theme_structure = theme_structure
        
        print("🔍 [QuestionAgent] Создание YandexGPT LLM...")
        self.llm = create_yandex_llm()
        print("✅ [QuestionAgent] YandexGPT LLM создан")
        
        self.current_theme_position = 0
        
        print("🔍 [QuestionAgent] Создание LangGraph состояний...")
        # Создаем граф состояний
        self.graph = self._create_question_graph()
        self.app = self.graph.compile()
        print("✅ [QuestionAgent] LangGraph граф создан и скомпилирован")
        
        print("🔍 [QuestionAgent] Настройка промптов...")
        self._setup_prompts()
        print("✅ [QuestionAgent] Агент полностью инициализирован")
    
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
            input_variables=["subject", "difficulty", "question_number", "topic_context", 
                           "previous_questions", "evaluation_characteristics"],
            template="""
Ты эксперт-экзаменатор по предмету "{subject}".

{topic_context}

Создай вопрос номер {question_number} уровня сложности "{difficulty}" строго по указанной теме.

ПРЕДЫДУЩИЕ ВОПРОСЫ:
{previous_questions}

ХАРАКТЕРИСТИКИ ПРЕДЫДУЩИХ ОТВЕТОВ:
{evaluation_characteristics}

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
            input_variables=["subject", "topic_context", "difficulty", "question_requirements", 
                           "evaluation_characteristics"],
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
    
    def _create_question_graph(self) -> StateGraph:
        """Создает граф состояний для генерации вопросов"""
        graph = StateGraph(QuestionState)
        
        # Добавляем узлы
        graph.add_node("determine_question_type", self._determine_question_type)
        graph.add_node("generate_initial_question", self._generate_initial_question_node)
        graph.add_node("generate_contextual_question", self._generate_contextual_question_node)
        graph.add_node("generate_theme_guided_question", self._generate_theme_guided_question_node)
        graph.add_node("parse_question_response", self._parse_question_response_node)
        graph.add_node("validate_question", self._validate_question_node)
        
        # Определяем точку входа
        graph.set_entry_point("determine_question_type")
        
        # Добавляем условные ребра
        graph.add_conditional_edges(
            "determine_question_type",
            self._decide_question_path,
            {
                "initial": "generate_initial_question",
                "contextual": "generate_contextual_question", 
                "theme_guided": "generate_theme_guided_question"
            }
        )
        
        # Добавляем ребра к парсингу
        graph.add_edge("generate_initial_question", "parse_question_response")
        graph.add_edge("generate_contextual_question", "parse_question_response")
        graph.add_edge("generate_theme_guided_question", "parse_question_response")
        
        # Добавляем ребро к валидации
        graph.add_edge("parse_question_response", "validate_question")
        
        # Завершение
        graph.add_edge("validate_question", END)
        
        return graph
    
    def _determine_question_type(self, state: QuestionState) -> QuestionState:
        """Определяет тип вопроса для генерации"""
        print(f"🔍 [QuestionAgent] Определение типа вопроса для вопроса #{state['question_number']}")
        try:
            # Логируем операцию
            self.log_operation("determine_question_type", state, None)
            
            # Если есть тематическая структура, используем её
            if self.theme_structure:
                state["question_type"] = "theme_guided"
                print("✅ [QuestionAgent] Выбран тип: theme_guided")
                return state
            
            # Если это первый вопрос
            if state["question_number"] == 1 or not state.get("evaluation_summaries"):
                state["question_type"] = "initial"
                print("✅ [QuestionAgent] Выбран тип: initial")
                return state
            
            # Иначе используем контекстный подход
            state["question_type"] = "contextual"
            print("✅ [QuestionAgent] Выбран тип: contextual")
            return state
            
        except Exception as e:
            error_msg = f"Ошибка определения типа вопроса: {str(e)}"
            print(f"❌ [QuestionAgent] {error_msg}")
            state["error"] = error_msg
            self.log_operation("determine_question_type", state, None, str(e))
            return state
    
    def _decide_question_path(self, state: QuestionState) -> str:
        """Решает, какой путь использовать для генерации"""
        if state.get("error"):
            return "initial"  # Fallback к начальному вопросу
        
        return state.get("question_type", "initial")
    
    def _generate_initial_question_node(self, state: QuestionState) -> QuestionState:
        """Генерирует первый вопрос"""
        print(f"🔍 [QuestionAgent] Генерация первого вопроса для предмета: {self.subject}")
        try:
            print("🔍 [QuestionAgent] Создание цепочки промпт → LLM → парсер...")
            chain = self.initial_question_prompt | self.llm | StrOutputParser()
            
            print("🔍 [QuestionAgent] Подготовка данных для промпта...")
            prompt_data = {
                "subject": self.subject,
                "difficulty": self.difficulty,
                "topic_context": self.topic_context
            }
            print(f"🔍 [QuestionAgent] Данные промпта: {prompt_data}")
            
            print("🔍 [QuestionAgent] Вызов YandexGPT API...")
            response = chain.invoke(prompt_data)
            
            print(f"✅ [QuestionAgent] Получен ответ от YandexGPT: {response[:100]}...")
            state["raw_response"] = response
            self.log_operation("generate_initial_question", state, response)
            return state
            
        except Exception as e:
            error_msg = f"Ошибка генерации начального вопроса: {str(e)}"
            print(f"❌ [QuestionAgent] {error_msg}")
            state["error"] = error_msg
            self.log_operation("generate_initial_question", state, None, str(e))
            return state
    
    def _generate_contextual_question_node(self, state: QuestionState) -> QuestionState:
        """Генерирует контекстный вопрос"""
        try:
            # Форматируем предыдущие вопросы
            previous_questions = self._format_previous_questions()
            
            # Форматируем характеристики оценок
            evaluation_characteristics = self._format_evaluation_characteristics(
                state.get("evaluation_summaries", [])
            )
            
            chain = self.contextual_question_prompt | self.llm | StrOutputParser()
            
            response = chain.invoke({
                "subject": self.subject,
                "difficulty": self.difficulty,
                "question_number": state["question_number"],
                "topic_context": self.topic_context,
                "previous_questions": previous_questions,
                "evaluation_characteristics": evaluation_characteristics
            })
            
            state["raw_response"] = response
            self.log_operation("generate_contextual_question", state, response)
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка генерации контекстного вопроса: {str(e)}"
            self.log_operation("generate_contextual_question", state, None, str(e))
            return state
    
    def _generate_theme_guided_question_node(self, state: QuestionState) -> QuestionState:
        """Генерирует вопрос на основе тематической структуры"""
        try:
            # Получаем требования для следующего вопроса
            theme_requirements = self._get_next_question_requirements()
            
            if not theme_requirements or theme_requirements.get("error"):
                # Fallback к контекстному вопросу
                return self._generate_contextual_question_node(state)
            
            # Форматируем требования и характеристики
            requirements_text = self._format_requirements_for_prompt(theme_requirements)
            evaluation_characteristics = self._format_evaluation_characteristics(
                state.get("evaluation_summaries", [])
            )
            
            chain = self.theme_guided_question_prompt | self.llm | StrOutputParser()
            
            response = chain.invoke({
                "subject": self.subject,
                "topic_context": self.topic_context,
                "difficulty": self.difficulty,
                "question_requirements": requirements_text,
                "evaluation_characteristics": evaluation_characteristics
            })
            
            state["raw_response"] = response
            state["theme_requirements"] = theme_requirements
            self.log_operation("generate_theme_guided_question", state, response)
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка генерации тематического вопроса: {str(e)}"
            self.log_operation("generate_theme_guided_question", state, None, str(e))
            return state
    
    def _parse_question_response_node(self, state: QuestionState) -> QuestionState:
        """Парсит ответ с вопросом"""
        try:
            if state.get("error") or not state.get("raw_response"):
                return state
            
            response = state["raw_response"]
            
            # Базовый парсинг
            question_match = re.search(r'ВОПРОС:\s*(.+?)(?=\n|$)', response, re.DOTALL)
            key_points_match = re.search(r'КЛЮЧЕВЫЕ_МОМЕНТЫ:\s*(.+?)(?=\n|$)', response)
            level_match = re.search(r'УРОВЕНЬ_ТЕМЫ:\s*(.+?)(?=\n|$)', response)
            reasoning_match = re.search(r'ОБОСНОВАНИЕ:\s*(.+?)(?=\n|$)', response, re.DOTALL)
            
            # Дополнительные поля для тематических вопросов
            bloom_level_match = re.search(r'УРОВЕНЬ_БЛУМА:\s*(.+?)(?=\n|$)', response)
            direction_match = re.search(r'ТЕМАТИЧЕСКОЕ_НАПРАВЛЕНИЕ:\s*(.+?)(?=\n|$)', response)
            process_match = re.search(r'КОГНИТИВНЫЙ_ПРОЦЕСС:\s*(.+?)(?=\n|$)', response)
            criteria_match = re.search(r'КРИТЕРИИ_ОЦЕНКИ:\s*(.+?)(?=\n|$)', response, re.DOTALL)
            adaptation_match = re.search(r'АДАПТАЦИЯ:\s*(.+?)(?=\n|$)', response, re.DOTALL)
            
            question_data = {
                'question': question_match.group(1).strip() if question_match else "Вопрос не найден",
                'key_points': key_points_match.group(1).strip() if key_points_match else "",
                'topic_level': level_match.group(1).strip() if level_match else "базовый",
                'reasoning': reasoning_match.group(1).strip() if reasoning_match else "",
                'question_number': state["question_number"],
                'timestamp': datetime.now(),
                'raw_response': response
            }
            
            # Добавляем поля для тематических вопросов
            if bloom_level_match or state.get("theme_requirements"):
                theme_requirements = state.get("theme_requirements", {})
                question_data.update({
                    'bloom_level': theme_requirements.get('bloom_level', 'remember'),
                    'bloom_level_name': theme_requirements.get('level_name', 'Не указан'),
                    'thematic_direction': direction_match.group(1).strip() if direction_match else "",
                    'cognitive_process': process_match.group(1).strip() if process_match else "",
                    'evaluation_criteria': criteria_match.group(1).strip() if criteria_match else "",
                    'adaptation_notes': adaptation_match.group(1).strip() if adaptation_match else "",
                    'theme_requirements': theme_requirements,
                    'topic_level': self._map_bloom_to_topic_level(theme_requirements.get('bloom_level', 'remember'))
                })
            
            state["generated_question"] = question_data
            self.log_operation("parse_question_response", response, question_data)
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка парсинга вопроса: {str(e)}"
            self.log_operation("parse_question_response", state, None, str(e))
            return state
    
    def _validate_question_node(self, state: QuestionState) -> QuestionState:
        """Валидирует сгенерированный вопрос"""
        try:
            if state.get("error"):
                return state
            
            question_data = state.get("generated_question")
            if not question_data:
                state["error"] = "Нет данных вопроса для валидации"
                return state
            
            # Базовая валидация
            validation_errors = []
            
            if not question_data.get("question") or question_data["question"] == "Вопрос не найден":
                validation_errors.append("Не удалось извлечь текст вопроса")
            
            if len(question_data.get("question", "")) < 10:
                validation_errors.append("Вопрос слишком короткий")
            
            if not question_data.get("key_points"):
                validation_errors.append("Отсутствуют ключевые моменты")
            
            # Проверка на повторение (если есть история)
            if hasattr(self, 'question_history'):
                for prev_q in self.question_history:
                    if prev_q.get("question", "").lower() == question_data["question"].lower():
                        validation_errors.append("Вопрос повторяется")
                        break
            
            if validation_errors:
                state["error"] = f"Ошибки валидации: {'; '.join(validation_errors)}"
            else:
                # Обновляем позицию в тематической структуре
                if self.theme_structure:
                    self.current_theme_position += 1
                
                # Добавляем в историю
                if not hasattr(self, 'question_history'):
                    self.question_history = []
                self.question_history.append(question_data)
            
            self.log_operation("validate_question", question_data, validation_errors)
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка валидации вопроса: {str(e)}"
            self.log_operation("validate_question", state, None, str(e))
            return state
    
    def generate_question(self, question_number: int, evaluation_summaries: List[Dict] = None) -> Dict[str, Any]:
        """
        Генерирует вопрос с использованием LangGraph
        
        Args:
            question_number: Номер вопроса
            evaluation_summaries: Список характеристик оценок от EvaluationAgent
            
        Returns:
            Словарь с вопросом и метаданными
        """
        print(f"🔍 [QuestionAgent] Начало генерации вопроса #{question_number}")
        try:
            print("🔍 [QuestionAgent] Создание начального состояния...")
            # Создаем начальное состояние
            initial_state = QuestionState(
                question_number=question_number,
                evaluation_summaries=evaluation_summaries or [],
                theme_requirements=None,
                generated_question=None,
                error=None,
                raw_response=None, # Добавляем отсутствующее поле
                question_type=None # Добавляем отсутствующее поле
            )
            print(f"✅ [QuestionAgent] Состояние создано: номер вопроса {question_number}")
            
            print("🔍 [QuestionAgent] Запуск LangGraph workflow...")
            # Запускаем граф
            result = self.app.invoke(initial_state)
            print("✅ [QuestionAgent] LangGraph workflow завершен")
            
            # Проверяем результат
            if result.get("error"):
                error_msg = result["error"]
                print(f"❌ [QuestionAgent] Ошибка в workflow: {error_msg}")
                return {
                    "error": error_msg,
                    "question": "Ошибка генерации вопроса",
                    "key_points": "",
                    "question_number": question_number,
                    "timestamp": datetime.now()
                }
            
            print(f"✅ [QuestionAgent] Вопрос #{question_number} успешно сгенерирован")
            return result.get("generated_question", {})
            
        except Exception as e:
            error_msg = f"Критическая ошибка в generate_question: {str(e)}"
            print(f"❌ [QuestionAgent] {error_msg}")
            self.log_operation("generate_question", {"question_number": question_number}, None, error_msg)
            
            return {
                "error": error_msg,
                "question": "Ошибка генерации вопроса",
                "key_points": "",
                "question_number": question_number,
                "timestamp": datetime.now()
            }
    
    # Вспомогательные методы (скопированы из оригинального агента с адаптацией)
    
    def _format_previous_questions(self) -> str:
        """Форматирует предыдущие вопросы для промпта"""
        if not hasattr(self, 'question_history') or not self.question_history:
            return "Нет предыдущих вопросов"
        
        formatted = ""
        for i, q in enumerate(self.question_history, 1):
            formatted += f"{i}. {q['question']}\n"
        
        return formatted
    
    def _format_evaluation_characteristics(self, evaluation_summaries: List[Dict]) -> str:
        """Форматирует характеристики оценок БЕЗ текстов ответов студента"""
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
    
    def _get_next_question_requirements(self) -> Optional[Dict]:
        """Получает требования для следующего вопроса от ThemeAgent"""
        if not self.theme_structure:
            return None
        
        try:
            from theme_agent import ThemeAgentLangGraph
            theme_agent = ThemeAgentLangGraph()
            requirements = theme_agent.get_next_bloom_level_requirements(
                self.theme_structure, 
                self.current_theme_position
            )
            return requirements
        except:
            return None
    
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
    
    def get_question_history(self) -> List[Dict]:
        """Возвращает историю заданных вопросов"""
        return getattr(self, 'question_history', []).copy()
    
    def get_theme_progress(self) -> Dict[str, Any]:
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
        if hasattr(self, 'question_history'):
            self.question_history = []
        self.current_theme_position = 0
        super().reset_history()


# Функция для создания QuestionAgent на LangGraph
def create_question_agent(
    subject: str = "Общие знания",
    difficulty: str = "средний",
    topic_context: str = None,
    theme_structure: dict = None
) -> QuestionAgentLangGraph:
    """Создает экземпляр QuestionAgent на LangGraph"""
    print(f"🔍 [create_question_agent] Создание QuestionAgent для '{subject}' сложности '{difficulty}'")
    agent = QuestionAgentLangGraph(
        subject=subject,
        difficulty=difficulty,
        topic_context=topic_context,
        theme_structure=theme_structure
    )
    print("✅ [create_question_agent] QuestionAgent создан успешно")
    return agent

# Псевдоним для обратной совместимости
def create_question_agent_langgraph(
    subject: str = "Общие знания",
    difficulty: str = "средний",
    topic_context: str = None,
    theme_structure: dict = None
) -> QuestionAgentLangGraph:
    """Создает экземпляр QuestionAgent на LangGraph (псевдоним для обратной совместимости)"""
    return create_question_agent(subject, difficulty, topic_context, theme_structure)

