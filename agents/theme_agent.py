"""
Агент для создания тематической структуры экзамена на основе таксономии Блума на LangGraph
"""
from typing import Dict, List, Optional, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from base import LangGraphAgentBase, ThemeState
from yagpt_llm import create_yandex_llm
import json
import re
from datetime import datetime


class ThemeAgentLangGraph(LangGraphAgentBase):
    """Агент для создания тематической структуры экзамена с руководящими принципами для QuestionAgent на LangGraph"""
    
    def __init__(self, subject: str = "Общие знания", topic_context: str = None):
        super().__init__(subject, topic_context)
        self.llm = create_yandex_llm()
        self.generated_structures = []
        
        # Структура таксономии Блума (пересмотренная версия)
        self.bloom_levels = {
            "remember": {
                "name": "Запоминание",
                "description": "Извлечение знаний из долговременной памяти",
                "verbs": ["определить", "назвать", "перечислить", "описать", "узнать", "вспомнить"],
                "question_types": ["Что такое...?", "Назовите...", "Перечислите...", "Определите..."],
                "weight": 0.15  # 15% от общего количества вопросов
            },
            "understand": {
                "name": "Понимание", 
                "description": "Понимание значения материала",
                "verbs": ["объяснить", "интерпретировать", "сравнить", "классифицировать", "обобщить"],
                "question_types": ["Объясните...", "Почему...?", "В чем разница...?", "Сравните..."],
                "weight": 0.25  # 25% от общего количества вопросов
            },
            "apply": {
                "name": "Применение",
                "description": "Использование знаний в новых ситуациях",
                "verbs": ["использовать", "решить", "показать", "продемонстрировать", "применить"],
                "question_types": ["Как бы вы использовали...?", "Решите задачу...", "Покажите как..."],
                "weight": 0.25  # 25% от общего количества вопросов
            },
            "analyze": {
                "name": "Анализ",
                "description": "Разделение на части и понимание связей",
                "verbs": ["анализировать", "различать", "исследовать", "сравнить", "противопоставить"],
                "question_types": ["Проанализируйте...", "Какие факторы...?", "Какие доказательства...?"],
                "weight": 0.20  # 20% от общего количества вопросов
            },
            "evaluate": {
                "name": "Оценивание",
                "description": "Формирование суждений на основе критериев",
                "verbs": ["оценить", "критиковать", "судить", "обосновать", "аргументировать"],
                "question_types": ["Оцените...", "Что лучше...?", "Обоснуйте...", "Критически оцените..."],
                "weight": 0.10  # 10% от общего количества вопросов
            },
            "create": {
                "name": "Создание",
                "description": "Создание нового продукта или точки зрения",
                "verbs": ["создать", "разработать", "построить", "спланировать", "произвести"],
                "question_types": ["Создайте...", "Разработайте план...", "Предложите решение..."],
                "weight": 0.05  # 5% от общего количества вопросов
            }
        }
        
        # Создаем граф состояний
        self.graph = self._create_theme_graph()
        self.app = self.graph.compile()
        
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Настройка промптов для создания тематической структуры"""
        
        # Промпт для анализа темы и создания структуры обучения
        self.theme_analysis_prompt = PromptTemplate(
            input_variables=["topic_context", "bloom_levels_info"],
            template="""
Ты эксперт по педагогическому дизайну и таксономии Блума.
Проанализируй тему и создай структуру обучения и оценки по всем уровням таксономии Блума.

ТЕМА ЭКЗАМЕНА:
{topic_context}

УРОВНИ ТАКСОНОМИИ БЛУМА:
{bloom_levels_info}

ЗАДАЧА:
Создай тематическую структуру обучения, определив для каждого уровня Блума:
1. Ключевые концепции и аспекты темы для изучения
2. Руководящие принципы для создания вопросов
3. Ожидаемые результаты обучения
4. Критерии оценки

ВАЖНО: НЕ создавай конкретные вопросы! Создавай только структуру и принципы.

ФОРМАТ ОТВЕТА:
ОБЩИЙ_ОБЗОР: [краткое описание тематической структуры]

УРОВЕНЬ_ЗАПОМИНАНИЕ:
Ключевые_концепции: [основные понятия, термины, факты для запоминания]
Руководящие_принципы: [как должны формулироваться вопросы этого уровня]
Примеры_направлений: [примеры тем для вопросов, НЕ сами вопросы]
Критерии_оценки: [как оценивать ответы на этом уровне]

УРОВЕНЬ_ПОНИМАНИЕ:
Ключевые_концепции: [концепции для понимания и объяснения]
Руководящие_принципы: [принципы формулирования вопросов]
Примеры_направлений: [направления для создания вопросов]
Критерии_оценки: [критерии оценки понимания]

УРОВЕНЬ_ПРИМЕНЕНИЕ:
Ключевые_концепции: [навыки и знания для практического применения]
Руководящие_принципы: [как создавать практические задания]
Примеры_направлений: [области для практических заданий]
Критерии_оценки: [оценка практических навыков]

УРОВЕНЬ_АНАЛИЗ:
Ключевые_концепции: [элементы для анализа и исследования]
Руководящие_принципы: [принципы создания аналитических заданий]
Примеры_направлений: [направления для анализа]
Критерии_оценки: [оценка аналитических способностей]

УРОВЕНЬ_ОЦЕНИВАНИЕ:
Ключевые_концепции: [аспекты для критической оценки]
Руководящие_принципы: [как формулировать оценочные задания]
Примеры_направлений: [области для критической оценки]
Критерии_оценки: [оценка критического мышления]

УРОВЕНЬ_СОЗДАНИЕ:
Ключевые_концепции: [элементы для творческого синтеза]
Руководящие_принципы: [принципы создания творческих заданий]
Примеры_направлений: [направления для творческих проектов]
Критерии_оценки: [оценка творческих способностей]

РЕКОМЕНДУЕМАЯ_ПОСЛЕДОВАТЕЛЬНОСТЬ: [в каком порядке изучать уровни]
МЕЖУРОВНЕВЫЕ_СВЯЗИ: [как уровни связаны между собой]
"""
        )
        
        # Промпт для создания конкретных руководящих принципов для QuestionAgent
        self.question_guidelines_prompt = PromptTemplate(
            input_variables=["topic_context", "bloom_level", "level_structure", "question_count"],
            template="""
Ты эксперт по созданию образовательных материалов.
Создай подробные руководящие принципы для генерации {question_count} вопросов уровня "{bloom_level}".

ТЕМА:
{topic_context}

СТРУКТУРА УРОВНЯ:
{level_structure}

ЗАДАЧА:
Создай руководящие принципы для QuestionAgent, чтобы он мог сгенерировать качественные вопросы.
НЕ создавай сами вопросы - только детальные инструкции для их создания.

ФОРМАТ ОТВЕТА:
ПРИНЦИПЫ_ФОРМУЛИРОВАНИЯ:
[как должны формулироваться вопросы этого уровня]

ОБЯЗАТЕЛЬНЫЕ_ЭЛЕМЕНТЫ:
[что обязательно должно быть в каждом вопросе]

ТЕМАТИЧЕСКИЕ_НАПРАВЛЕНИЯ:
[конкретные аспекты темы для вопросов]

УРОВЕНЬ_СЛОЖНОСТИ:
[требования к сложности формулировок]

КОНТЕКСТНЫЕ_ТРЕБОВАНИЯ:
[как связать вопросы с темой]

КРИТЕРИИ_КАЧЕСТВА:
[как понять, что вопрос хорошо сформулирован]

ИЗБЕГАТЬ:
[чего не должно быть в вопросах этого уровня]
"""
        )
    
    def _create_theme_graph(self) -> StateGraph:
        """Создает граф состояний для создания тематической структуры"""
        graph = StateGraph(ThemeState)
        
        # Добавляем узлы
        graph.add_node("validate_theme_input", self._validate_theme_input_node)
        graph.add_node("format_bloom_levels", self._format_bloom_levels_node)
        graph.add_node("analyze_theme", self._analyze_theme_node)
        graph.add_node("distribute_questions", self._distribute_questions_node)
        graph.add_node("create_question_guidelines", self._create_question_guidelines_node)
        graph.add_node("build_theme_curriculum", self._build_theme_curriculum_node)
        graph.add_node("validate_theme_structure", self._validate_theme_structure_node)
        graph.add_node("save_theme_structure", self._save_theme_structure_node)
        
        # Определяем точку входа
        graph.set_entry_point("validate_theme_input")
        
        # Добавляем последовательные ребра
        graph.add_edge("validate_theme_input", "format_bloom_levels")
        graph.add_edge("format_bloom_levels", "analyze_theme")
        graph.add_edge("analyze_theme", "distribute_questions")
        graph.add_edge("distribute_questions", "create_question_guidelines")
        graph.add_edge("create_question_guidelines", "build_theme_curriculum")
        graph.add_edge("build_theme_curriculum", "validate_theme_structure")
        graph.add_edge("validate_theme_structure", "save_theme_structure")
        
        # Завершение
        graph.add_edge("save_theme_structure", END)
        
        return graph
    
    def _validate_theme_input_node(self, state: ThemeState) -> ThemeState:
        """Валидирует входные данные для создания тематической структуры"""
        try:
            if not state.get("subject"):
                state["error"] = "Отсутствует предмет"
                return state
            
            if not state.get("topic_context"):
                state["error"] = "Отсутствует контекст темы"
                return state
            
            if state.get("total_questions", 0) <= 0:
                state["error"] = "Некорректное количество вопросов"
                return state
            
            self.log_operation("validate_theme_input", state, "Validation passed")
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка валидации входных данных: {str(e)}"
            self.log_operation("validate_theme_input", state, None, str(e))
            return state
    
    def _format_bloom_levels_node(self, state: ThemeState) -> ThemeState:
        """Форматирует информацию об уровнях Блума для промпта"""
        try:
            if state.get("error"):
                return state
            
            info = ""
            for level, data in self.bloom_levels.items():
                info += f"\n{data['name'].upper()} ({level}):\n"
                info += f"Описание: {data['description']}\n"
                info += f"Действия: {', '.join(data['verbs'])}\n"
                info += f"Типы вопросов: {', '.join(data['question_types'])}\n"
                info += f"Доля от экзамена: {int(data['weight']*100)}%\n"
            
            state["bloom_levels_info"] = info
            self.log_operation("format_bloom_levels", len(self.bloom_levels), len(info))
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка форматирования уровней Блума: {str(e)}"
            self.log_operation("format_bloom_levels", state, None, str(e))
            return state
    
    def _analyze_theme_node(self, state: ThemeState) -> ThemeState:
        """Анализирует тему и создает структуру обучения"""
        try:
            if state.get("error"):
                return state
            
            chain = self.theme_analysis_prompt | self.llm | StrOutputParser()
            
            theme_structure = chain.invoke({
                "topic_context": state["topic_context"],
                "bloom_levels_info": state["bloom_levels_info"]
            })
            
            state["raw_theme_structure"] = theme_structure
            self.log_operation("analyze_theme", state["topic_context"][:100], theme_structure[:200])
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка анализа темы: {str(e)}"
            self.log_operation("analyze_theme", state, None, str(e))
            return state
    
    def _distribute_questions_node(self, state: ThemeState) -> ThemeState:
        """Распределяет вопросы по уровням Блума согласно весам"""
        try:
            if state.get("error"):
                return state
            
            total_questions = state["total_questions"]
            distribution = {}
            
            for level, data in self.bloom_levels.items():
                count = max(1, round(total_questions * data['weight']))
                distribution[level] = count
            
            # Корректировка для точного соответствия общему количеству
            current_total = sum(distribution.values())
            if current_total != total_questions:
                # Добавляем или убираем вопросы с наиболее важных уровней
                diff = total_questions - current_total
                priority_levels = ['understand', 'apply', 'analyze']
                
                for level in priority_levels:
                    if diff == 0:
                        break
                    if diff > 0:
                        distribution[level] += 1
                        diff -= 1
                    elif diff < 0 and distribution[level] > 1:
                        distribution[level] -= 1
                        diff += 1
            
            state["questions_distribution"] = distribution
            self.log_operation("distribute_questions", total_questions, distribution)
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка распределения вопросов: {str(e)}"
            self.log_operation("distribute_questions", state, None, str(e))
            return state
    
    def _create_question_guidelines_node(self, state: ThemeState) -> ThemeState:
        """Создает руководящие принципы для QuestionAgent по каждому уровню Блума"""
        try:
            if state.get("error"):
                return state
            
            distribution = state["questions_distribution"]
            raw_theme_structure = state["raw_theme_structure"]
            
            # Извлекаем структуру каждого уровня из общей структуры темы
            level_structures = self._parse_theme_structure(raw_theme_structure)
            
            guidelines = {}
            
            for level, count in distribution.items():
                if count > 0:
                    level_structure = level_structures.get(level, f"Структура для уровня {level}")
                    
                    chain = self.question_guidelines_prompt | self.llm | StrOutputParser()
                    
                    response = chain.invoke({
                        "topic_context": state["topic_context"],
                        "bloom_level": self.bloom_levels[level]['name'],
                        "level_structure": level_structure,
                        "question_count": count
                    })
                    
                    # Парсим и структурируем руководящие принципы
                    parsed_guidelines = self._parse_question_guidelines(response)
                    
                    guidelines[level] = {
                        'bloom_level': level,
                        'level_name': self.bloom_levels[level]['name'],
                        'question_count': count,
                        'guidelines': parsed_guidelines,
                        'raw_response': response
                    }
            
            state["question_guidelines"] = guidelines
            self.log_operation("create_question_guidelines", len(distribution), len(guidelines))
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка создания руководящих принципов: {str(e)}"
            self.log_operation("create_question_guidelines", state, None, str(e))
            return state
    
    def _build_theme_curriculum_node(self, state: ThemeState) -> ThemeState:
        """Строит итоговую тематическую структуру"""
        try:
            if state.get("error"):
                return state
            
            curriculum_id = self._generate_curriculum_id()
            
            theme_curriculum = {
                'curriculum_id': curriculum_id,
                'subject': state["subject"],
                'topic_context': state["topic_context"],
                'total_questions': state["total_questions"],
                'difficulty': state["difficulty"],
                'theme_structure': state["raw_theme_structure"],
                'questions_distribution': state["questions_distribution"],
                'question_guidelines': state["question_guidelines"],
                'bloom_sequence': self._create_bloom_sequence(),
                'assessment_framework': self._create_assessment_framework(),
                'metadata': {
                    'created_at': datetime.now(),
                    'bloom_coverage': self._calculate_bloom_coverage(state["questions_distribution"]),
                    'estimated_duration': self._estimate_duration(state["questions_distribution"])
                }
            }
            
            state["theme_structure"] = theme_curriculum
            self.log_operation("build_theme_curriculum", curriculum_id, "Curriculum built")
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка построения тематической структуры: {str(e)}"
            self.log_operation("build_theme_curriculum", state, None, str(e))
            return state
    
    def _validate_theme_structure_node(self, state: ThemeState) -> ThemeState:
        """Валидирует созданную тематическую структуру"""
        try:
            if state.get("error"):
                return state
            
            theme_structure = state["theme_structure"]
            validation_result = self._validate_structure(theme_structure)
            
            state["validation_result"] = validation_result
            
            if not validation_result.get("is_valid"):
                state["error"] = f"Структура не прошла валидацию: {validation_result.get('issues', [])}"
            
            self.log_operation("validate_theme_structure", "Structure validated", validation_result)
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка валидации тематической структуры: {str(e)}"
            self.log_operation("validate_theme_structure", state, None, str(e))
            return state
    
    def _save_theme_structure_node(self, state: ThemeState) -> ThemeState:
        """Сохраняет тематическую структуру в историю"""
        try:
            if state.get("error"):
                return state
            
            theme_structure = state["theme_structure"]
            if theme_structure:
                self.generated_structures.append(theme_structure)
                self.log_operation("save_theme_structure", "Structure saved", len(self.generated_structures))
            
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка сохранения тематической структуры: {str(e)}"
            self.log_operation("save_theme_structure", state, None, str(e))
            return state
    
    def _parse_theme_structure(self, theme_structure: str) -> Dict[str, str]:
        """Парсит структуру темы по уровням"""
        level_structures = {}
        
        # Паттерны для извлечения каждого уровня
        patterns = {
            'remember': r'УРОВЕНЬ_ЗАПОМИНАНИЕ:(.*?)(?=УРОВЕНЬ_|$)',
            'understand': r'УРОВЕНЬ_ПОНИМАНИЕ:(.*?)(?=УРОВЕНЬ_|$)',
            'apply': r'УРОВЕНЬ_ПРИМЕНЕНИЕ:(.*?)(?=УРОВЕНЬ_|$)',
            'analyze': r'УРОВЕНЬ_АНАЛИЗ:(.*?)(?=УРОВЕНЬ_|$)',
            'evaluate': r'УРОВЕНЬ_ОЦЕНИВАНИЕ:(.*?)(?=УРОВЕНЬ_|$)',
            'create': r'УРОВЕНЬ_СОЗДАНИЕ:(.*?)(?=УРОВЕНЬ_|$)'
        }
        
        for level, pattern in patterns.items():
            match = re.search(pattern, theme_structure, re.DOTALL)
            if match:
                level_structures[level] = match.group(1).strip()
            else:
                level_structures[level] = f"Базовая структура для уровня {level}"
        
        return level_structures
    
    def _parse_question_guidelines(self, response: str) -> Dict[str, str]:
        """Парсит руководящие принципы из ответа LLM"""
        guidelines = {}
        
        # Паттерны для извлечения различных секций
        patterns = {
            'formulation_principles': r'ПРИНЦИПЫ_ФОРМУЛИРОВАНИЯ:(.*?)(?=\n[А-Я_]+:|$)',
            'mandatory_elements': r'ОБЯЗАТЕЛЬНЫЕ_ЭЛЕМЕНТЫ:(.*?)(?=\n[А-Я_]+:|$)',
            'thematic_directions': r'ТЕМАТИЧЕСКИЕ_НАПРАВЛЕНИЯ:(.*?)(?=\n[А-Я_]+:|$)',
            'verbs_and_actions': r'ГЛАГОЛЫ_И_ДЕЙСТВИЯ:(.*?)(?=\n[А-Я_]+:|$)',
            'complexity_level': r'УРОВЕНЬ_СЛОЖНОСТИ:(.*?)(?=\n[А-Я_]+:|$)',
            'contextual_requirements': r'КОНТЕКСТНЫЕ_ТРЕБОВАНИЯ:(.*?)(?=\n[А-Я_]+:|$)',
            'quality_criteria': r'КРИТЕРИИ_КАЧЕСТВА:(.*?)(?=\n[А-Я_]+:|$)',
            'avoid': r'ИЗБЕГАТЬ:(.*?)(?=\n[А-Я_]+:|$)',
            'student_adaptation': r'АДАПТАЦИЯ_ПОД_СТУДЕНТА:(.*?)(?=\n[А-Я_]+:|$)',
            'format_requirements': r'ТРЕБОВАНИЯ_К_ФОРМАТУ:(.*?)(?=\n[А-Я_]+:|$)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.DOTALL)
            guidelines[key] = match.group(1).strip() if match else ""
        
        return guidelines
    
    def _create_bloom_sequence(self) -> List[str]:
        """Создает рекомендуемую последовательность уровней Блума"""
        return ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    
    def _create_assessment_framework(self) -> Dict[str, str]:
        """Создает общие принципы оценки для каждого уровня"""
        framework = {}
        
        for level, data in self.bloom_levels.items():
            if level == "remember":
                framework[level] = "Оценивайте точность воспроизведения фактов, терминов, определений"
            elif level == "understand":
                framework[level] = "Оценивайте способность объяснить смысл, интерпретировать, сравнивать"
            elif level == "apply":
                framework[level] = "Оценивайте умение использовать знания в новых ситуациях, решать практические задачи"
            elif level == "analyze":
                framework[level] = "Оценивайте способность разложить на части, выявить связи, найти доказательства"
            elif level == "evaluate":
                framework[level] = "Оценивайте качество аргументации, обоснованность суждений, критическое мышление"
            elif level == "create":
                framework[level] = "Оценивайте оригинальность, креативность, способность синтезировать новое"
        
        return framework
    
    def _calculate_bloom_coverage(self, distribution: Dict[str, int]) -> Dict[str, float]:
        """Вычисляет покрытие уровней Блума в процентах"""
        total = sum(distribution.values())
        return {level: (count / total) * 100 for level, count in distribution.items()}
    
    def _estimate_duration(self, distribution: Dict[str, int]) -> int:
        """Оценивает общую продолжительность экзамена в минутах (приблизительно)"""
        # Базовое приблизительное время для каждого уровня Блума
        level_time = {
            'remember': 3,      # простые вопросы на память
            'understand': 5,    # вопросы на понимание
            'apply': 8,         # практические задания
            'analyze': 10,      # аналитические задания
            'evaluate': 12,     # оценочные задания
            'create': 15        # творческие задания
        }
        
        total_time = 0
        for level, count in distribution.items():
            total_time += count * level_time.get(level, 5)
        
        # Добавляем время на инструктаж и переходы
        total_time += 10
        
        return total_time
    
    def _generate_curriculum_id(self) -> str:
        """Генерирует уникальный ID для программы"""
        return f"theme_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _validate_structure(self, curriculum: Dict[str, Any]) -> Dict[str, Any]:
        """Валидирует тематическую структуру экзамена"""
        issues = []
        warnings = []
        
        # Проверка наличия всех уровней Блума
        question_guidelines = curriculum.get('question_guidelines', {})
        missing_levels = set(self.bloom_levels.keys()) - set(question_guidelines.keys())
        
        if missing_levels:
            issues.append(f"Отсутствуют уровни Блума: {', '.join(missing_levels)}")
        
        # Проверка руководящих принципов
        for level, guidelines in question_guidelines.items():
            if not guidelines.get('guidelines', {}).get('formulation_principles'):
                warnings.append(f"Отсутствуют принципы формулирования для уровня {level}")
        
        # Проверка баланса вопросов
        distribution = curriculum.get('questions_distribution', {})
        total_questions = sum(distribution.values())
        
        if total_questions != curriculum.get('total_questions', 0):
            issues.append("Несоответствие общего количества вопросов и распределения")
        
        # Проверка временных рамок
        estimated_duration = curriculum.get('metadata', {}).get('estimated_duration', 0)
        if estimated_duration > 180:  # более 3 часов
            warnings.append("Экзамен может быть слишком длинным (>3 часов)")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'recommendations': self._generate_validation_recommendations(issues, warnings)
        }
    
    def _generate_validation_recommendations(self, issues: List[str], warnings: List[str]) -> List[str]:
        """Генерирует рекомендации на основе проблем валидации"""
        recommendations = []
        
        if issues:
            recommendations.append("Исправьте критические ошибки перед использованием структуры")
        
        if warnings:
            recommendations.append("Рассмотрите предупреждения для улучшения качества экзамена")
        
        if not issues and not warnings:
            recommendations.append("Тематическая структура готова к использованию с QuestionAgent")
        
        return recommendations
    
    def generate_theme_structure(self, total_questions: int = 10, difficulty: str = "средний") -> Dict[str, Any]:
        """
        Генерирует тематическую структуру экзамена с руководящими принципами с использованием LangGraph
        
        Args:
            total_questions: Общее количество вопросов
            difficulty: Уровень сложности экзамена
            
        Returns:
            Тематическая структура с принципами для QuestionAgent
        """
        try:
            # Создаем начальное состояние
            initial_state = ThemeState(
                subject=self.subject,
                topic_context=self.topic_context,
                total_questions=total_questions,
                difficulty=difficulty,
                theme_structure=None,
                validation_result=None,
                error=None
            )
            
            # Запускаем граф
            result = self.app.invoke(initial_state)
            
            # Проверяем результат
            if result.get("error"):
                return {'error': result["error"]}
            
            return result.get("theme_structure", {})
            
        except Exception as e:
            error_msg = f"Критическая ошибка в generate_theme_structure: {str(e)}"
            self.log_operation("generate_theme_structure", {
                "total_questions": total_questions,
                "difficulty": difficulty
            }, None, error_msg)
            
            return {'error': error_msg}
    
    def get_question_requirements_for_level(self, curriculum: Dict[str, Any], bloom_level: str) -> Dict[str, Any]:
        """
        Возвращает требования к вопросам для конкретного уровня Блума
        
        Args:
            curriculum: Тематическая структура
            bloom_level: Уровень Блума
            
        Returns:
            Требования для QuestionAgent
        """
        guidelines = curriculum.get('question_guidelines', {}).get(bloom_level, {})
        
        if not guidelines:
            return {'error': f'Нет руководящих принципов для уровня {bloom_level}'}
        
        return {
            'bloom_level': bloom_level,
            'level_name': guidelines.get('level_name'),
            'question_count': guidelines.get('question_count'),
            'formulation_principles': guidelines.get('guidelines', {}).get('formulation_principles'),
            'mandatory_elements': guidelines.get('guidelines', {}).get('mandatory_elements'),
            'thematic_directions': guidelines.get('guidelines', {}).get('thematic_directions'),
            'verbs_and_actions': guidelines.get('guidelines', {}).get('verbs_and_actions'),
            'complexity_level': guidelines.get('guidelines', {}).get('complexity_level'),
            'quality_criteria': guidelines.get('guidelines', {}).get('quality_criteria'),
            'contextual_requirements': guidelines.get('guidelines', {}).get('contextual_requirements'),
            'avoid': guidelines.get('guidelines', {}).get('avoid'),
            'student_adaptation': guidelines.get('guidelines', {}).get('student_adaptation'),
            'format_requirements': guidelines.get('guidelines', {}).get('format_requirements')
        }
    
    def get_next_bloom_level_requirements(self, curriculum: Dict[str, Any], current_position: int) -> Optional[Dict]:
        """
        Возвращает требования для следующего вопроса согласно последовательности Блума
        
        Args:
            curriculum: Тематическая структура
            current_position: Текущая позиция в экзамене (0-based)
            
        Returns:
            Требования для следующего вопроса или None если экзамен завершен
        """
        distribution = curriculum.get('questions_distribution', {})
        bloom_sequence = curriculum.get('bloom_sequence', ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create'])
        
        # Считаем общее количество вопросов и определяем текущий уровень
        position_counter = 0
        
        for level in bloom_sequence:
            level_questions = distribution.get(level, 0)
            if current_position < position_counter + level_questions:
                # Мы находимся на этом уровне
                return self.get_question_requirements_for_level(curriculum, level)
            position_counter += level_questions
        
        return None  # Экзамен завершен
    
    def generate_summary_report(self, curriculum: Dict[str, Any]) -> str:
        """
        Генерирует итоговый отчет о тематической структуре экзамена
        
        Args:
            curriculum: Тематическая структура
            
        Returns:
            Текстовый отчет
        """
        report = f"""
=== ОТЧЕТ О ТЕМАТИЧЕСКОЙ СТРУКТУРЕ ЭКЗАМЕНА ===

ОБЩАЯ ИНФОРМАЦИЯ:
• Предмет: {curriculum['subject']}
• Общее количество вопросов: {curriculum['total_questions']}
• Уровень сложности: {curriculum['difficulty']}
• Примерная продолжительность: {curriculum['metadata']['estimated_duration']} минут

РАСПРЕДЕЛЕНИЕ ПО УРОВНЯМ БЛУМА:
"""
        
        distribution = curriculum['questions_distribution']
        coverage = curriculum['metadata']['bloom_coverage']
        
        for level, count in distribution.items():
            level_name = self.bloom_levels[level]['name']
            percentage = coverage[level]
            report += f"• {level_name}: {count} вопросов ({percentage:.1f}%)\n"
        
        report += f"""
ОСОБЕННОСТИ ТЕМАТИЧЕСКОЙ СТРУКТУРЫ:
• QuestionAgent получает детальные руководящие принципы для каждого уровня
• Вопросы генерируются динамически на основе требований ThemeAgent
• Обеспечивается адаптация под конкретную тему и уровень студента
• Поддерживается последовательное развитие когнитивных навыков

РЕКОМЕНДАЦИИ ПО ПРОВЕДЕНИЮ:
• Следуйте рекомендуемой последовательности уровней Блума
• Используйте руководящие принципы для контроля качества вопросов
• Адаптируйте сложность в зависимости от ответов студента
• Обращайте внимание на межуровневые связи в обучении
"""
        
        return report
    
    def get_structure_history(self) -> List[Dict]:
        """Возвращает историю созданных структур"""
        return self.generated_structures.copy()
    
    def export_structure_to_json(self, curriculum: Dict[str, Any]) -> str:
        """Экспортирует тематическую структуру в JSON формат"""
        return json.dumps(curriculum, ensure_ascii=False, indent=2, default=str)
    
    def validate_theme_structure(self, curriculum: Dict[str, Any]) -> Dict[str, Any]:
        """
        Валидирует тематическую структуру экзамена
        
        Args:
            curriculum: Структура для валидации
            
        Returns:
            Результат валидации
        """
        return self._validate_structure(curriculum)
    
    def reset_history(self):
        """Сбрасывает историю созданных структур"""
        self.generated_structures = []
        super().reset_history()


# Функция для создания ThemeAgent на LangGraph
def create_theme_agent(
    subject: str = "Общие знания",
    topic_context: str = None
) -> ThemeAgentLangGraph:
    """Создает экземпляр ThemeAgent на LangGraph"""
    return ThemeAgentLangGraph(
        subject=subject,
        topic_context=topic_context
    )

# Псевдоним для обратной совместимости
def create_theme_agent_langgraph(
    subject: str = "Общие знания",
    topic_context: str = None
) -> ThemeAgentLangGraph:
    """Создает экземпляр ThemeAgent на LangGraph (псевдоним для обратной совместимости)"""
    return create_theme_agent(subject, topic_context)
