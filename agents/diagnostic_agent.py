"""
Агент для диагностики и финальной оценки экзамена на LangGraph
"""
from typing import Dict, List, Optional, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from base import LangGraphAgentBase, DiagnosticState
from yagpt_llm import create_yandex_llm
import json
import re
import statistics
from datetime import datetime


class DiagnosticAgentLangGraph(LangGraphAgentBase):
    """Агент для комплексной диагностики результатов экзамена на LangGraph"""
    
    def __init__(self, subject: str = "Общие знания", topic_context: str = None):
        print(f"🔍 [DiagnosticAgent] Инициализация агента диагностики для предмета: {subject}")
        super().__init__(subject, topic_context)
        
        print("🔍 [DiagnosticAgent] Создание YandexGPT LLM...")
        self.llm = create_yandex_llm()
        print("✅ [DiagnosticAgent] YandexGPT LLM создан")
        
        self.diagnostic_history = []
        
        print("🔍 [DiagnosticAgent] Создание LangGraph состояний...")
        # Создаем граф состояний
        self.graph = self._create_diagnostic_graph()
        self.app = self.graph.compile()
        print("✅ [DiagnosticAgent] LangGraph граф создан и скомпилирован")
        
        print("🔍 [DiagnosticAgent] Настройка промптов...")
        self._setup_prompts()
        print("✅ [DiagnosticAgent] Агент полностью инициализирован")
    
    def _setup_prompts(self):
        """Настройка промптов для диагностики"""
        
        # Промпт для анализа паттернов ответов
        self.pattern_analysis_prompt = PromptTemplate(
            input_variables=["subject", "topic_context", "questions_and_evaluations", "overall_stats"],
            template="""
Ты эксперт-диагност образовательного процесса по предмету "{subject}".

{topic_context}

Проанализируй паттерны в ответах студента по конкретной теме и выяви ключевые проблемы и сильные стороны именно в этой области.

ДАННЫЕ ОБ ОТВЕТАХ:
{questions_and_evaluations}

ОБЩАЯ СТАТИСТИКА:
{overall_stats}

ЗАДАЧИ АНАЛИЗА:
1. Выяви ПАТТЕРНЫ в ошибках и успехах студента
2. Определи ПРОБЕЛЫ в знаниях по конкретным темам
3. Оцени ПРОГРЕСС в процессе экзамена (улучшение/ухудшение)
4. Выяви КОГНИТИВНЫЕ ОСОБЕННОСТИ (логическое мышление, память, понимание)
5. Определи СТИЛЬ ОБУЧЕНИЯ студента

ФОРМАТ АНАЛИЗА:
ПАТТЕРНЫ_ОШИБОК: [типичные ошибки и их причины]
СИЛЬНЫЕ_СТОРОНЫ: [в чем студент особенно силен]
ПРОБЕЛЫ_ЗНАНИЙ: [конкретные темы, требующие изучения]
КОГНИТИВНЫЙ_ПРОФИЛЬ: [особенности мышления студента]
ПРОГРЕСС_ДИНАМИКА: [как менялись ответы в процессе экзамена]
СТИЛЬ_ОБУЧЕНИЯ: [рекомендации по оптимальному способу обучения]
КРИТИЧЕСКИЕ_ОБЛАСТИ: [самые проблемные зоны, требующие немедленного внимания]

Будь конкретен и основывайся только на данных из ответов.
"""
        )
        
        # Промпт для финального отчета
        self.final_report_prompt = PromptTemplate(
            input_variables=["subject", "pattern_analysis", "total_score", "max_score", "grade_recommendation"],
            template="""
Составь итоговый диагностический отчет об экзамене студента по предмету "{subject}".

АНАЛИЗ ПАТТЕРНОВ:
{pattern_analysis}

ИТОГОВЫЕ БАЛЛЫ: {total_score} из {max_score} ({grade_recommendation})

Создай КОМПЛЕКСНЫЙ ОТЧЕТ, включающий:

1. ИСПОЛНИТЕЛЬСКОЕ РЕЗЮМЕ (краткая сводка результатов)
2. ДЕТАЛЬНАЯ ДИАГНОСТИКА ЗНАНИЙ
3. ПРОФИЛЬ ОБУЧАЮЩЕГОСЯ (сильные/слабые стороны)
4. РЕКОМЕНДАЦИИ ПО ОБУЧЕНИЮ (конкретные шаги)
5. ПЛАН УСТРАНЕНИЯ ПРОБЕЛОВ (приоритетные действия)
6. ПРОГНОЗ РАЗВИТИЯ (потенциал студента)

ТРЕБОВАНИЯ К ОТЧЕТУ:
- Будь объективен и конструктивен
- Предоставь КОНКРЕТНЫЕ рекомендации, а не общие фразы
- Укажи ПРИОРИТЕТЫ в обучении
- Дай МОТИВИРУЮЩУЮ обратную связь
- Используй данные для обоснования выводов

СТРУКТУРА:
=== ИСПОЛНИТЕЛЬСКОЕ РЕЗЮМЕ ===
[краткие выводы и итоговая оценка]

=== ДИАГНОСТИКА ЗНАНИЙ ===
[детальный анализ уровня владения предметом]

=== ПРОФИЛЬ ОБУЧАЮЩЕГОСЯ ===
[индивидуальные особенности студента]

=== РЕКОМЕНДАЦИИ ===
[конкретные шаги для улучшения]

=== ПЛАН ДЕЙСТВИЙ ===
[приоритетные задачи с временными рамками]

=== ПРОГНОЗ ===
[потенциал и ожидаемое развитие]
"""
        )
        
        # Промпт для сравнительного анализа
        self.comparative_analysis_prompt = PromptTemplate(
            input_variables=["current_results", "benchmark_data"],
            template="""
Сравни результаты студента с эталонными данными и нормами.

РЕЗУЛЬТАТЫ СТУДЕНТА:
{current_results}

ЭТАЛОННЫЕ ДАННЫЕ:
{benchmark_data}

Проведи сравнительный анализ и дай рекомендации по позиционированию студента.

ФОРМАТ:
ПОЗИЦИЯ_ОТНОСИТЕЛЬНО_НОРМЫ: [выше/на уровне/ниже среднего]
СИЛЬНЫЕ_ОБЛАСТИ_В_СРАВНЕНИИ: [где студент превосходит норму]
ОТСТАЮЩИЕ_ОБЛАСТИ: [где студент уступает норме]
РЕКОМЕНДАЦИИ_ПО_РАЗВИТИЮ: [как достичь нормативного уровня]
"""
        )
    
    def _create_diagnostic_graph(self) -> StateGraph:
        """Создает граф состояний для диагностики"""
        graph = StateGraph(DiagnosticState)
        
        # Добавляем узлы
        graph.add_node("validate_diagnostic_input", self._validate_diagnostic_input_node)
        graph.add_node("prepare_analysis_data", self._prepare_analysis_data_node)
        graph.add_node("analyze_patterns", self._analyze_patterns_node)
        graph.add_node("calculate_statistics", self._calculate_statistics_node)
        graph.add_node("determine_grade", self._determine_grade_node)
        graph.add_node("generate_final_report", self._generate_final_report_node)
        graph.add_node("extract_recommendations", self._extract_recommendations_node)
        graph.add_node("identify_critical_areas", self._identify_critical_areas_node)
        graph.add_node("create_diagnostic_result", self._create_diagnostic_result_node)
        graph.add_node("save_diagnostic_history", self._save_diagnostic_history_node)
        
        # Определяем точку входа
        graph.set_entry_point("validate_diagnostic_input")
        
        # Добавляем последовательные ребра
        graph.add_edge("validate_diagnostic_input", "prepare_analysis_data")
        graph.add_edge("prepare_analysis_data", "analyze_patterns")
        graph.add_edge("analyze_patterns", "calculate_statistics")
        graph.add_edge("calculate_statistics", "determine_grade")
        graph.add_edge("determine_grade", "generate_final_report")
        graph.add_edge("generate_final_report", "extract_recommendations")
        graph.add_edge("extract_recommendations", "identify_critical_areas")
        graph.add_edge("identify_critical_areas", "create_diagnostic_result")
        graph.add_edge("create_diagnostic_result", "save_diagnostic_history")
        
        # Завершение
        graph.add_edge("save_diagnostic_history", END)
        
        return graph
    
    def _validate_diagnostic_input_node(self, state: DiagnosticState) -> DiagnosticState:
        """Валидирует входные данные для диагностики"""
        try:
            questions = state.get("questions", [])
            evaluations = state.get("evaluations", [])
            
            if not questions or not evaluations:
                state["error"] = 'Недостаточно данных для диагностики'
                return state
            
            if len(questions) != len(evaluations):
                state["error"] = 'Несоответствие количества вопросов и оценок'
                return state
            
            self.log_operation("validate_diagnostic_input", state, "Validation passed")
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка валидации входных данных: {str(e)}"
            self.log_operation("validate_diagnostic_input", state, None, str(e))
            return state
    
    def _prepare_analysis_data_node(self, state: DiagnosticState) -> DiagnosticState:
        """Подготавливает данные для анализа"""
        try:
            if state.get("error"):
                return state
            
            questions = state["questions"]
            evaluations = state["evaluations"]
            
            analysis_text = ""
            
            for i, (question, evaluation) in enumerate(zip(questions, evaluations), 1):
                analysis_text += f"\n--- ВОПРОС {i} ---\n"
                analysis_text += f"Вопрос: {question.get('question', 'Не указан')}\n"
                analysis_text += f"Уровень сложности: {question.get('topic_level', 'Не указан')}\n"
                analysis_text += f"Ключевые моменты: {question.get('key_points', 'Не указаны')}\n"
                
                if evaluation.get('type') == 'detailed':
                    eval_data = evaluation
                    analysis_text += f"Итоговая оценка: {eval_data.get('total_score', 0)}/10\n"
                    analysis_text += f"Оценки по критериям:\n"
                    criteria = eval_data.get('criteria_scores', {})
                    for criterion, score in criteria.items():
                        analysis_text += f"  - {criterion}: {score}/10\n"
                    analysis_text += f"Сильные стороны: {eval_data.get('strengths', '')}\n"
                    analysis_text += f"Области улучшения: {eval_data.get('weaknesses', '')}\n"
                else:
                    analysis_text += f"Оценка: {evaluation.get('total_score', 0)}/10\n"
                    analysis_text += f"Комментарий: {evaluation.get('comment', '')}\n"
                
                analysis_text += "\n"
            
            state["analysis_data"] = analysis_text
            self.log_operation("prepare_analysis_data", len(questions), len(analysis_text))
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка подготовки данных: {str(e)}"
            self.log_operation("prepare_analysis_data", state, None, str(e))
            return state
    
    def _analyze_patterns_node(self, state: DiagnosticState) -> DiagnosticState:
        """Анализирует паттерны в ответах"""
        try:
            if state.get("error"):
                return state
            
            analysis_data = state["analysis_data"]
            questions = state["questions"]
            
            # Вычисление общей статистики для контекста
            stats_text = f"Общее количество вопросов: {len(questions)}\n"
            
            chain = self.pattern_analysis_prompt | self.llm | StrOutputParser()
            
            pattern_analysis = chain.invoke({
                "subject": self.subject,
                "topic_context": self.topic_context,
                "questions_and_evaluations": analysis_data,
                "overall_stats": stats_text
            })
            
            state["pattern_analysis"] = pattern_analysis
            self.log_operation("analyze_patterns", analysis_data[:200], pattern_analysis[:200])
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка анализа паттернов: {str(e)}"
            self.log_operation("analyze_patterns", state, None, str(e))
            return state
    
    def _calculate_statistics_node(self, state: DiagnosticState) -> DiagnosticState:
        """Вычисляет статистики по оценкам"""
        try:
            if state.get("error"):
                return state
            
            evaluations = state["evaluations"]
            scores = []
            detailed_scores = {'correctness': [], 'completeness': [], 'understanding': [], 'structure': []}
            
            for evaluation in evaluations:
                score = evaluation.get('total_score', 0)
                scores.append(score)
                
                # Детальные критерии (если доступны)
                if evaluation.get('type') == 'detailed':
                    criteria = evaluation.get('criteria_scores', {})
                    for criterion, value in criteria.items():
                        if criterion in detailed_scores:
                            detailed_scores[criterion].append(value)
            
            total_score = sum(scores)
            max_score = len(scores) * 10
            average_score = total_score / len(scores) if scores else 0
            
            stats = {
                'total_score': total_score,
                'max_score': max_score,
                'average_score': round(average_score, 2),
                'percentage': round((total_score / max_score) * 100, 1) if max_score > 0 else 0,
                'individual_scores': scores,
                'highest_score': max(scores) if scores else 0,
                'lowest_score': min(scores) if scores else 0
            }
            
            # Добавление статистик по критериям
            for criterion, criterion_scores in detailed_scores.items():
                if criterion_scores:
                    stats[f'{criterion}_average'] = round(sum(criterion_scores) / len(criterion_scores), 2)
            
            # Анализ распределения
            if scores:
                stats['score_distribution'] = {
                    'excellent': len([s for s in scores if s >= 9]),
                    'good': len([s for s in scores if 7 <= s < 9]),
                    'satisfactory': len([s for s in scores if 5 <= s < 7]),
                    'poor': len([s for s in scores if s < 5])
                }
                
                # Тренд (если более 2 оценок)
                if len(scores) >= 3:
                    first_half = scores[:len(scores)//2]
                    second_half = scores[len(scores)//2:]
                    avg_first = sum(first_half) / len(first_half)
                    avg_second = sum(second_half) / len(second_half)
                    stats['trend'] = 'улучшение' if avg_second > avg_first else 'ухудшение' if avg_second < avg_first else 'стабильно'
            
            state["statistics"] = stats
            self.log_operation("calculate_statistics", len(evaluations), stats)
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка вычисления статистики: {str(e)}"
            self.log_operation("calculate_statistics", state, None, str(e))
            return state
    
    def _determine_grade_node(self, state: DiagnosticState) -> DiagnosticState:
        """Определяет итоговую оценку"""
        try:
            if state.get("error"):
                return state
            
            stats = state["statistics"]
            total_score = stats['total_score']
            max_score = stats['max_score']
            
            if max_score == 0:
                grade_info = {'grade': 'неопределено', 'percentage': 0, 'description': 'Нет данных для оценки'}
            else:
                percentage = (total_score / max_score) * 100
                
                if percentage >= 90:
                    grade = 'отлично'
                    description = 'Выдающееся владение материалом'
                elif percentage >= 75:
                    grade = 'хорошо'
                    description = 'Хорошее понимание предмета с небольшими пробелами'
                elif percentage >= 60:
                    grade = 'удовлетворительно'
                    description = 'Базовое понимание предмета, требуется дополнительное изучение'
                elif percentage >= 40:
                    grade = 'неудовлетворительно'
                    description = 'Серьезные пробелы в знаниях, требуется переподготовка'
                else:
                    grade = 'критически низко'
                    description = 'Критически низкий уровень знаний, требуется полное переобучение'
                
                grade_info = {
                    'grade': grade,
                    'percentage': round(percentage, 1),
                    'description': description,
                    'points': f"{total_score}/{max_score}"
                }
            
            state["grade_info"] = grade_info
            self.log_operation("determine_grade", stats, grade_info)
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка определения оценки: {str(e)}"
            self.log_operation("determine_grade", state, None, str(e))
            return state
    
    def _generate_final_report_node(self, state: DiagnosticState) -> DiagnosticState:
        """Генерирует финальный отчет"""
        try:
            if state.get("error"):
                return state
            
            pattern_analysis = state["pattern_analysis"]
            stats = state["statistics"]
            grade_info = state["grade_info"]
            
            chain = self.final_report_prompt | self.llm | StrOutputParser()
            
            final_report = chain.invoke({
                "subject": self.subject,
                "pattern_analysis": pattern_analysis,
                "total_score": stats['total_score'],
                "max_score": stats['max_score'],
                "grade_recommendation": f"{grade_info['grade']} ({grade_info['percentage']}%)"
            })
            
            state["final_report"] = final_report
            self.log_operation("generate_final_report", pattern_analysis[:100], final_report[:200])
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка генерации финального отчета: {str(e)}"
            self.log_operation("generate_final_report", state, None, str(e))
            return state
    
    def _extract_recommendations_node(self, state: DiagnosticState) -> DiagnosticState:
        """Извлекает рекомендации из отчета"""
        try:
            if state.get("error"):
                return state
            
            final_report = state["final_report"]
            
            # Ищем секцию рекомендаций
            recommendations_section = re.search(r'=== РЕКОМЕНДАЦИИ ===(.+?)(?==== |$)', final_report, re.DOTALL)
            
            recommendations = []
            if recommendations_section:
                text = recommendations_section.group(1).strip()
                # Разбиваем на отдельные рекомендации
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('===') and len(line) > 10:
                        # Убираем маркеры списков
                        clean_line = re.sub(r'^[-*•]\s*', '', line)
                        clean_line = re.sub(r'^\d+\.\s*', '', clean_line)
                        if clean_line:
                            recommendations.append(clean_line)
            
            # Если не нашли структурированные рекомендации, создаем базовые
            if not recommendations:
                recommendations = [
                    "Продолжить изучение основных концепций по теме",
                    "Практиковаться в решении задач разного уровня сложности",
                    "Обратить внимание на слабые места в понимании",
                    "Повторить материал по ключевым темам",
                    "Проконсультироваться с преподавателем по сложным вопросам"
                ]
            
            state["recommendations"] = recommendations[:8]  # Ограничиваем количество
            self.log_operation("extract_recommendations", final_report[:100], len(recommendations))
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка извлечения рекомендаций: {str(e)}"
            self.log_operation("extract_recommendations", state, None, str(e))
            return state
    
    def _identify_critical_areas_node(self, state: DiagnosticState) -> DiagnosticState:
        """Выявляет критические области для улучшения"""
        try:
            if state.get("error"):
                return state
            
            analysis_data = state["analysis_data"]
            critical_areas = []
            
            # Ищем паттерны низких оценок
            low_score_pattern = re.findall(r'Итоговая оценка: ([0-4])/10', analysis_data)
            if len(low_score_pattern) >= 2:
                critical_areas.append("Критически низкие оценки по большинству вопросов")
            
            # Ищем повторяющиеся проблемы в областях улучшения
            improvement_areas = re.findall(r'(?:Области улучшения|Слабые стороны): (.+)', analysis_data)
            
            # Анализируем темы проблем
            problem_themes = {
                'понимание': ['понимание', 'понимает', 'непонимание', 'неправильное понимание'],
                'полнота ответа': ['неполно', 'не полностью', 'отсутствие деталей', 'поверхностно'],
                'конкретность': ['неконкретно', 'отсутствие примеров', 'абстрактно', 'нет конкретных'],
                'структура': ['неструктурированно', 'хаотично', 'без логики', 'бессвязно'],
                'знания': ['ошибки в фактах', 'неточности', 'неверная информация']
            }
            
            # Подсчитываем упоминания тем
            theme_counts = {}
            for area in improvement_areas:
                area_lower = area.lower()
                for theme, keywords in problem_themes.items():
                    for keyword in keywords:
                        if keyword in area_lower:
                            theme_counts[theme] = theme_counts.get(theme, 0) + 1
                            break
            
            # Добавляем часто упоминаемые темы проблем
            for theme, count in theme_counts.items():
                if count >= 2:
                    critical_areas.append(f"Систематические проблемы с: {theme}")
            
            # Также ищем точные повторения
            exact_areas = {}
            for area in improvement_areas:
                cleaned_area = area.strip()
                if len(cleaned_area) > 15:  # Игнорируем слишком короткие фразы
                    exact_areas[cleaned_area] = exact_areas.get(cleaned_area, 0) + 1
            
            for area, count in exact_areas.items():
                if count >= 2:
                    critical_areas.append(f"Повторяющаяся проблема: {area}")
            
            if not critical_areas:
                critical_areas.append("Критические области не выявлены")
            
            state["critical_areas"] = critical_areas[:5]  # Ограничиваем количество
            self.log_operation("identify_critical_areas", analysis_data[:100], len(critical_areas))
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка выявления критических областей: {str(e)}"
            self.log_operation("identify_critical_areas", state, None, str(e))
            return state
    
    def _create_diagnostic_result_node(self, state: DiagnosticState) -> DiagnosticState:
        """Создает итоговый результат диагностики"""
        try:
            if state.get("error"):
                return state
            
            diagnostic_result = {
                'subject': self.subject,
                'pattern_analysis': state["pattern_analysis"],
                'statistics': state["statistics"],
                'grade_info': state["grade_info"],
                'final_report': state["final_report"],
                'recommendations': state["recommendations"],
                'critical_areas': state["critical_areas"],
                'timestamp': datetime.now()
            }
            
            state["diagnostic_result"] = diagnostic_result
            self.log_operation("create_diagnostic_result", "All components", "Result created")
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка создания результата диагностики: {str(e)}"
            self.log_operation("create_diagnostic_result", state, None, str(e))
            return state
    
    def _save_diagnostic_history_node(self, state: DiagnosticState) -> DiagnosticState:
        """Сохраняет результат в историю диагностик"""
        try:
            if state.get("error"):
                return state
            
            diagnostic_result = state["diagnostic_result"]
            if diagnostic_result:
                self.diagnostic_history.append(diagnostic_result)
                self.log_operation("save_diagnostic_history", "Result saved", len(self.diagnostic_history))
            
            return state
            
        except Exception as e:
            state["error"] = f"Ошибка сохранения в историю: {str(e)}"
            self.log_operation("save_diagnostic_history", state, None, str(e))
            return state
    
    def diagnose_exam_results(self, questions: List[Dict], evaluations: List[Dict], 
                            detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        Проводит комплексную диагностику результатов экзамена с использованием LangGraph
        
        Args:
            questions: Список вопросов с метаданными
            evaluations: Список оценок ответов
            detailed_analysis: Использовать детальный анализ
            
        Returns:
            Диагностический отчет
        """
        print(f"🔍 [DiagnosticAgent] Начало диагностики: {len(questions)} вопросов, {len(evaluations)} оценок")
        try:
            # Создаем начальное состояние
            initial_state = DiagnosticState(
                questions=questions,
                evaluations=evaluations,
                diagnostic_result=None,
                error=None
            )
            
            print("🔍 [DiagnosticAgent] Запуск LangGraph workflow для диагностики...")
            # Запускаем граф
            result = self.app.invoke(initial_state)
            
            # Проверяем результат
            if result.get("error"):
                print(f"❌ [DiagnosticAgent] Ошибка в workflow: {result['error']}")
                return {'error': result["error"]}
            
            print("✅ [DiagnosticAgent] Диагностика завершена успешно")
            return result.get("diagnostic_result", {})
            
        except Exception as e:
            error_msg = f"Критическая ошибка в diagnose_exam_results: {str(e)}"
            print(f"❌ [DiagnosticAgent] {error_msg}")
            self.log_operation("diagnose_exam_results", {
                "questions_count": len(questions),
                "evaluations_count": len(evaluations)
            }, None, error_msg)
            
            return {'error': error_msg}
    
    def run_diagnostics(self, questions: List[Dict], evaluations: List[Dict], 
                       detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        Алиас для diagnose_exam_results для обратной совместимости
        
        Args:
            questions: Список вопросов с метаданными
            evaluations: Список оценок ответов
            detailed_analysis: Использовать детальный анализ
            
        Returns:
            Диагностический отчет
        """
        print("🔍 [DiagnosticAgent] Вызван run_diagnostics (алиас для diagnose_exam_results)")
        return self.diagnose_exam_results(questions, evaluations, detailed_analysis)
    
    def compare_with_benchmark(self, current_results: Dict, benchmark_data: Dict = None) -> Dict[str, Any]:
        """Сравнивает результаты с эталонными данными"""
        try:
            if not benchmark_data:
                # Используем стандартные нормы
                benchmark_data = {
                    'average_score': 7.0,
                    'excellence_threshold': 9.0,
                    'passing_threshold': 6.0,
                    'typical_distribution': {
                        'excellent': 0.15,
                        'good': 0.35,
                        'satisfactory': 0.35,
                        'poor': 0.15
                    }
                }
            
            chain = self.comparative_analysis_prompt | self.llm | StrOutputParser()
            
            comparison = chain.invoke({
                "current_results": str(current_results),
                "benchmark_data": str(benchmark_data)
            })
            
            return {
                'comparison_analysis': comparison,
                'benchmark_data': benchmark_data,
                'performance_level': self._determine_performance_level(
                    current_results.get('average_score', 0),
                    benchmark_data.get('average_score', 7.0)
                )
            }
            
        except Exception as e:
            self.log_operation("compare_with_benchmark", current_results, None, str(e))
            return {'error': f"Ошибка сравнения с эталоном: {str(e)}"}
    
    def _determine_performance_level(self, student_avg: float, benchmark_avg: float) -> str:
        """Определяет уровень успеваемости относительно эталона"""
        ratio = student_avg / benchmark_avg if benchmark_avg > 0 else 0
        
        if ratio >= 1.2:
            return "значительно выше среднего"
        elif ratio >= 1.1:
            return "выше среднего"
        elif ratio >= 0.9:
            return "на уровне среднего"
        elif ratio >= 0.8:
            return "ниже среднего"
        else:
            return "значительно ниже среднего"
    
    def get_diagnostic_history(self) -> List[Dict]:
        """Возвращает историю диагностик"""
        return self.diagnostic_history.copy()
    
    def generate_learning_roadmap(self, diagnostic_result: Dict) -> Dict[str, Any]:
        """Генерирует дорожную карту обучения"""
        try:
            recommendations = diagnostic_result.get('recommendations', [])
            critical_areas = diagnostic_result.get('critical_areas', [])
            
            # Простая дорожная карта на основе рекомендаций
            roadmap = {
                'immediate_actions': critical_areas[:3],
                'short_term_goals': recommendations[:3],
                'medium_term_goals': recommendations[3:6] if len(recommendations) > 3 else [],
                'long_term_goals': recommendations[6:] if len(recommendations) > 6 else []
            }
            
            return roadmap
            
        except Exception as e:
            self.log_operation("generate_learning_roadmap", diagnostic_result, None, str(e))
            return {'error': f"Ошибка создания дорожной карты: {str(e)}"}
    
    def reset_history(self):
        """Сбрасывает историю диагностик"""
        self.diagnostic_history = []
        super().reset_history()


# Функция для создания DiagnosticAgent на LangGraph
def create_diagnostic_agent(
    subject: str = "Общие знания",
    topic_context: str = None
) -> DiagnosticAgentLangGraph:
    """Создает экземпляр DiagnosticAgent на LangGraph"""
    print(f"🔍 [create_diagnostic_agent] Создание DiagnosticAgent для '{subject}'")
    agent = DiagnosticAgentLangGraph(
        subject=subject,
        topic_context=topic_context
    )
    print("✅ [create_diagnostic_agent] DiagnosticAgent создан успешно")
    return agent

# Псевдоним для обратной совместимости
def create_diagnostic_agent_langgraph(
    subject: str = "Общие знания",
    topic_context: str = None
) -> DiagnosticAgentLangGraph:
    """Создает экземпляр DiagnosticAgent на LangGraph (псевдоним для обратной совместимости)"""
    return create_diagnostic_agent(subject, topic_context)
