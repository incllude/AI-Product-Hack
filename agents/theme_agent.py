"""
Агент для создания тематической структуры экзамена на основе таксономии Блума
"""
from typing import Dict, List, Optional, Tuple
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from yagpt_llm import YandexGPT
import json
import re


class ThemeAgent:
    """Агент для создания тематической структуры экзамена с руководящими принципами для QuestionAgent"""
    
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
        
        self.generated_structures = []
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
    
    def generate_theme_structure(self, total_questions: int = 10, difficulty: str = "средний") -> Dict[str, any]:
        """
        Генерирует тематическую структуру экзамена с руководящими принципами
        
        Args:
            total_questions: Общее количество вопросов
            difficulty: Уровень сложности экзамена
            
        Returns:
            Тематическая структура с принципами для QuestionAgent
        """
        # Формирование информации об уровнях Блума
        bloom_info = self._format_bloom_levels_info()
        
        # Анализ темы и создание общей структуры
        theme_structure = self._analyze_theme_and_create_structure(bloom_info)
        
        # Распределение вопросов по уровням
        questions_distribution = self._distribute_questions(total_questions)
        
        # Создание руководящих принципов для каждого уровня
        question_guidelines = self._create_question_guidelines(questions_distribution, theme_structure)
        
        # Создание итоговой структуры
        theme_curriculum = {
            'curriculum_id': self._generate_curriculum_id(),
            'subject': self.subject,
            'topic_context': self.topic_context,
            'total_questions': total_questions,
            'difficulty': difficulty,
            'theme_structure': theme_structure,
            'questions_distribution': questions_distribution,
            'question_guidelines': question_guidelines,
            'bloom_sequence': self._create_bloom_sequence(),
            'assessment_framework': self._create_assessment_framework(),
            'metadata': {
                'created_at': None,  # Можно добавить datetime
                'bloom_coverage': self._calculate_bloom_coverage(questions_distribution),
                'estimated_duration': self._estimate_duration(questions_distribution)
            }
        }
        
        # Сохранение в историю
        self.generated_structures.append(theme_curriculum)
        
        return theme_curriculum
    
    def _format_bloom_levels_info(self) -> str:
        """Форматирует информацию об уровнях Блума для промпта"""
        info = ""
        for level, data in self.bloom_levels.items():
            info += f"\n{data['name'].upper()} ({level}):\n"
            info += f"Описание: {data['description']}\n"
            info += f"Действия: {', '.join(data['verbs'])}\n"
            info += f"Типы вопросов: {', '.join(data['question_types'])}\n"
            info += f"Доля от экзамена: {int(data['weight']*100)}%\n"
        return info
    
    def _analyze_theme_and_create_structure(self, bloom_info: str) -> str:
        """Анализирует тему и создает структуру обучения"""
        chain = LLMChain(llm=self.llm, prompt=self.theme_analysis_prompt)
        
        return chain.run(
            topic_context=self.topic_context,
            bloom_levels_info=bloom_info
        )
    
    def _distribute_questions(self, total_questions: int) -> Dict[str, int]:
        """Распределяет вопросы по уровням Блума согласно весам"""
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
        
        return distribution
    
    def _create_question_guidelines(self, distribution: Dict[str, int], theme_structure: str) -> Dict[str, Dict]:
        """Создает руководящие принципы для QuestionAgent по каждому уровню Блума"""
        guidelines = {}
        
        # Извлекаем структуру каждого уровня из общей структуры темы
        level_structures = self._parse_theme_structure(theme_structure)
        
        for level, count in distribution.items():
            if count > 0:
                level_structure = level_structures.get(level, f"Структура для уровня {level}")
                
                chain = LLMChain(llm=self.llm, prompt=self.question_guidelines_prompt)
                
                response = chain.run(
                    topic_context=self.topic_context,
                    bloom_level=self.bloom_levels[level]['name'],
                    level_structure=level_structure,
                    question_count=count
                )
                
                # Парсим и структурируем руководящие принципы
                parsed_guidelines = self._parse_question_guidelines(response)
                
                guidelines[level] = {
                    'bloom_level': level,
                    'level_name': self.bloom_levels[level]['name'],
                    'question_count': count,
                    'guidelines': parsed_guidelines,
                    'raw_response': response
                }
        
        return guidelines
    
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
        # Поскольку таймеры убраны, оценка основана на сложности уровней Блума
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
        
        # Примечание: это только ориентировочная оценка, так как таймеры не используются
        return total_time
    
    def _generate_curriculum_id(self) -> str:
        """Генерирует уникальный ID для программы"""
        import datetime
        return f"theme_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_question_requirements_for_level(self, curriculum: Dict[str, any], bloom_level: str) -> Dict[str, any]:
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
    
    def get_next_bloom_level_requirements(self, curriculum: Dict[str, any], current_position: int) -> Optional[Dict]:
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
    
    def generate_summary_report(self, curriculum: Dict[str, any]) -> str:
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
    
    def export_structure_to_json(self, curriculum: Dict[str, any]) -> str:
        """Экспортирует тематическую структуру в JSON формат"""
        return json.dumps(curriculum, ensure_ascii=False, indent=2)
    
    def validate_theme_structure(self, curriculum: Dict[str, any]) -> Dict[str, any]:
        """
        Валидирует тематическую структуру экзамена
        
        Args:
            curriculum: Структура для валидации
            
        Returns:
            Результат валидации
        """
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