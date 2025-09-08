"""
Агент для диагностики и финальной оценки экзамена
"""
from typing import Dict, List, Optional, Tuple
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from yagpt_llm import YandexGPT
import json
import re
import statistics


class DiagnosticAgent:
    """Агент для комплексной диагностики результатов экзамена"""
    
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
        self.diagnostic_history = []
        
        self._setup_prompts()
    
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
    
    def diagnose_exam_results(self, questions: List[Dict], evaluations: List[Dict], 
                            detailed_analysis: bool = True) -> Dict[str, any]:
        """
        Проводит комплексную диагностику результатов экзамена
        
        Args:
            questions: Список вопросов с метаданными
            evaluations: Список оценок ответов
            detailed_analysis: Использовать детальный анализ
            
        Returns:
            Диагностический отчет
        """
        if not questions or not evaluations:
            return {'error': 'Недостаточно данных для диагностики'}
        
        # Подготовка данных для анализа
        analysis_data = self._prepare_analysis_data(questions, evaluations)
        
        # Анализ паттернов
        pattern_analysis = self._analyze_patterns(analysis_data)
        
        # Вычисление статистик
        stats = self._calculate_statistics(evaluations)
        
        # Определение оценки
        grade_info = self._determine_grade(stats['total_score'], stats['max_score'])
        
        # Создание финального отчета
        final_report = self._generate_final_report(
            pattern_analysis, stats, grade_info
        )
        
        diagnostic_result = {
            'subject': self.subject,
            'pattern_analysis': pattern_analysis,
            'statistics': stats,
            'grade_info': grade_info,
            'final_report': final_report,
            'recommendations': self._extract_recommendations(final_report),
            'critical_areas': self._identify_critical_areas(analysis_data),
            'timestamp': None  # Можно добавить datetime
        }
        
        # Сохранение в историю
        self.diagnostic_history.append(diagnostic_result)
        
        return diagnostic_result
    
    def _prepare_analysis_data(self, questions: List[Dict], evaluations: List[Dict]) -> str:
        """Подготавливает данные для анализа"""
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
                analysis_text += f"Области улучшения: {eval_data.get('areas_for_improvement', '')}\n"
            else:
                analysis_text += f"Оценка: {evaluation.get('total_score', 0)}/10\n"
                analysis_text += f"Комментарий: {evaluation.get('comment', '')}\n"
            
            analysis_text += "\n"
        
        return analysis_text
    
    def _analyze_patterns(self, analysis_data: str) -> str:
        """Анализирует паттерны в ответах"""
        # Вычисление общей статистики для контекста
        stats_text = f"Общее количество вопросов: {len(analysis_data.split('--- ВОПРОС'))}\n"
        
        chain = LLMChain(llm=self.llm, prompt=self.pattern_analysis_prompt)
        
        return chain.run(
            subject=self.subject,
            topic_context=self.topic_context,
            questions_and_evaluations=analysis_data,
            overall_stats=stats_text
        )
    
    def _calculate_statistics(self, evaluations: List[Dict]) -> Dict[str, any]:
        """Вычисляет статистики по оценкам"""
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
        
        return stats
    
    def _determine_grade(self, total_score: float, max_score: float) -> Dict[str, any]:
        """Определяет итоговую оценку"""
        if max_score == 0:
            return {'grade': 'неопределено', 'percentage': 0, 'description': 'Нет данных для оценки'}
        
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
        
        return {
            'grade': grade,
            'percentage': round(percentage, 1),
            'description': description,
            'points': f"{total_score}/{max_score}"
        }
    
    def _generate_final_report(self, pattern_analysis: str, stats: Dict, grade_info: Dict) -> str:
        """Генерирует финальный отчет"""
        chain = LLMChain(llm=self.llm, prompt=self.final_report_prompt)
        
        return chain.run(
            subject=self.subject,
            pattern_analysis=pattern_analysis,
            total_score=stats['total_score'],
            max_score=stats['max_score'],
            grade_recommendation=f"{grade_info['grade']} ({grade_info['percentage']}%)"
        )
    
    def _extract_recommendations(self, final_report: str) -> List[str]:
        """Извлекает рекомендации из отчета"""
        recommendations_section = re.search(r'=== РЕКОМЕНДАЦИИ ===\s*(.+?)(?===|\Z)', final_report, re.DOTALL)
        
        if not recommendations_section:
            return ["Рекомендации не найдены в отчете"]
        
        recommendations_text = recommendations_section.group(1).strip()
        
        # Разбиваем на отдельные рекомендации
        recommendations = []
        for line in recommendations_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('==='):
                # Убираем маркеры списка
                line = re.sub(r'^[-•*]\s*', '', line)
                if line:
                    recommendations.append(line)
        
        return recommendations[:10]  # Ограничиваем количество
    
    def _identify_critical_areas(self, analysis_data: str) -> List[str]:
        """Выявляет критические области для улучшения"""
        critical_areas = []
        
        # Ищем паттерны низких оценок
        low_score_pattern = re.findall(r'Итоговая оценка: ([0-4])/10', analysis_data)
        if len(low_score_pattern) >= 2:
            critical_areas.append("Критически низкие оценки по большинству вопросов")
        
        # Ищем повторяющиеся проблемы в областях улучшения
        improvement_areas = re.findall(r'Области улучшения: (.+)', analysis_data)
        common_words = {}
        for area in improvement_areas:
            words = area.lower().split()
            for word in words:
                if len(word) > 4:  # Игнорируем короткие слова
                    common_words[word] = common_words.get(word, 0) + 1
        
        # Добавляем часто упоминаемые проблемы
        for word, count in common_words.items():
            if count >= 2:
                critical_areas.append(f"Повторяющиеся проблемы с: {word}")
        
        if not critical_areas:
            critical_areas.append("Критические области не выявлены")
        
        return critical_areas[:5]  # Ограничиваем количество
    
    def _extract_recommendations(self, final_report: str) -> List[str]:
        """Извлекает рекомендации из финального отчета"""
        recommendations = []
        
        # Ищем секцию рекомендаций
        recommendations_section = re.search(r'=== РЕКОМЕНДАЦИИ ===(.+?)(?==== |$)', final_report, re.DOTALL)
        
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
        
        return recommendations[:8]  # Ограничиваем количество
    
    def compare_with_benchmark(self, current_results: Dict, benchmark_data: Dict = None) -> Dict[str, any]:
        """Сравнивает результаты с эталонными данными"""
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
        
        chain = LLMChain(llm=self.llm, prompt=self.comparative_analysis_prompt)
        
        comparison = chain.run(
            current_results=str(current_results),
            benchmark_data=str(benchmark_data)
        )
        
        return {
            'comparison_analysis': comparison,
            'benchmark_data': benchmark_data,
            'performance_level': self._determine_performance_level(
                current_results.get('average_score', 0),
                benchmark_data.get('average_score', 7.0)
            )
        }
    
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
    
    def generate_learning_roadmap(self, diagnostic_result: Dict) -> Dict[str, any]:
        """Генерирует дорожную карту обучения"""
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
    
    def reset_history(self):
        """Сбрасывает историю диагностик"""
        self.diagnostic_history = []