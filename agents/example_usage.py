"""
Пример использования системы экзаменационных агентов на LangGraph
Демонстрирует все возможности новой архитектуры
"""
import sys
import os
import time
from datetime import datetime

# Добавляем текущую директорию в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exam_orchestrator import create_exam_orchestrator_langgraph as create_exam_orchestrator
from exam_workflow import create_exam_workflow as create_exam_workflow
from question_agent import create_question_agent as create_question_agent
from evaluation_agent import create_evaluation_agent as create_evaluation_agent
from diagnostic_agent import create_diagnostic_agent as create_diagnostic_agent
from theme_agent import create_theme_agent as create_theme_agent
from yagpt_llm import validate_yandex_config
from base import create_initial_exam_state, ExamSession
from topic_manager import TopicManager


def main_example():
    """Основной пример использования LangGraph системы"""
    print("=== СИСТЕМА ЭКЗАМЕНИРОВАНИЯ НА LANGGRAPH ===\n")
    
    # Проверяем конфигурацию YandexGPT
    config = validate_yandex_config()
    if not config['is_valid']:
        print("⚠️  ВНИМАНИЕ: YandexGPT не настроен!")
        print(f"   Отсутствуют переменные: {config.get('missing_vars', [])}")
        print("   Система будет работать в демонстрационном режиме.\n")
    else:
        print("✅ YandexGPT настроен корректно\n")
    
    # Выбор темы экзамена
    topic_manager = TopicManager()
    topic_info = topic_manager.get_topic_selection()
    
    print(f"\n📚 Выбранная тема: {topic_info['name']}")
    print(f"🎓 Предмет: {topic_info['subject']}")
    print(f"📝 Описание: {topic_info['description']}")
    print(f"⚙️  Сложность: {topic_info['difficulty']}")
    
    # Создание LangGraph оркестратора
    print("\n🚀 Создание LangGraph оркестратора...")
    orchestrator = create_exam_orchestrator(
        topic_info=topic_info,
        max_questions=5,
        use_theme_structure=True  # Используем тематическую структуру
    )
    
    print("\n🎯 Система готова к проведению экзамена\n")
    
    # Демонстрация экзамена
    simulate_langgraph_exam(orchestrator, topic_info, config['is_valid'])


def simulate_langgraph_exam(orchestrator, topic_info: dict, use_real_llm: bool = True):
    """Симуляция экзамена через LangGraph оркестратор"""
    
    # Примеры ответов в зависимости от темы
    if "python" in topic_info['name'].lower() or "цикл" in topic_info['name'].lower():
        sample_answers = [
            "Цикл for используется для перебора элементов последовательности, а while выполняется пока условие истинно. For удобнее для известного количества итераций.",
            "Break прерывает выполнение цикла полностью, а continue пропускает текущую итерацию и переходит к следующей.",
            "numbers = [1, 2, 3, 4, 5]\nfor i in range(len(numbers)):\n    print(f'Элемент {i}: {numbers[i]}')",
            "List comprehension - это способ создания списков в одну строку. Например: [x*2 for x in range(5)] создаст [0, 2, 4, 6, 8]",
            "Вложенные циклы - это циклы внутри других циклов. Используются для работы с многомерными структурами данных, например матрицами."
        ]
    elif "фотоэффект" in topic_info['name'].lower() or "физик" in topic_info['subject'].lower():
        sample_answers = [
            "Фотоэффект - это явление испускания электронов веществом под действием света. Открыт Герцем, объяснен Эйнштейном через квантовую природу света.",
            "Уравнение Эйнштейна: E = hν = A + Ek, где hν - энергия фотона, A - работа выхода, Ek - кинетическая энергия электрона.",
            "Красная граница - это минимальная частота света, при которой еще возможен фотоэффект. Определяется работой выхода: ν₀ = A/h",
            "Законы фотоэффекта: 1) Количество электронов пропорционально интенсивности света 2) Кинетическая энергия не зависит от интенсивности 3) Есть красная граница",
            "Квантовая природа проявляется в том, что электрон поглощает энергию фотона целиком, а не постепенно. Это объясняет независимость энергии от интенсивности."
        ]
    else:
        sample_answers = [
            "Это основной принцип или концепция в данной области, который требует понимания фундаментальных основ.",
            "Здесь важно учитывать взаимосвязь различных элементов и их влияние на общую систему.",
            "Практическое применение этого принципа можно увидеть в реальных примерах и задачах.",
            "Данная концепция имеет несколько аспектов, каждый из которых важен для полного понимания.",
            "Современные подходы к этой проблеме учитывают последние достижения в данной области."
        ]
    
    # Запуск экзамена
    print("🚀 Запускаю экзамен через LangGraph оркестратор...")
    session_info = orchestrator.start_exam("LangGraph Демо-студент")
    print(f"   Сессия: {session_info['session_id']}")
    
    # Показываем информацию о тематической структуре
    if session_info.get('use_theme_structure'):
        theme_info = orchestrator.get_theme_structure_info()
        if not theme_info.get('error'):
            print(f"\n🧠 ТЕМАТИЧЕСКАЯ СТРУКТУРА СОЗДАНА:")
            print(f"   📊 Всего вопросов: {theme_info['total_questions']}")
            print(f"   ⏱️  Время экзамена: {theme_info['estimated_duration']} минут")
            
            distribution = theme_info.get('questions_distribution', {})
            bloom_names = {
                'remember': 'Запоминание', 'understand': 'Понимание', 'apply': 'Применение',
                'analyze': 'Анализ', 'evaluate': 'Оценивание', 'create': 'Создание'
            }
            
            print(f"   📋 Распределение по уровням Блума:")
            for level, count in distribution.items():
                name = bloom_names.get(level, level)
                print(f"      🔹 {name}: {count} вопросов")
    
    # Проведение экзамена
    for i in range(orchestrator.max_questions):
        print(f"\n{'='*70}")
        print(f"ВОПРОС {i + 1} ИЗ {orchestrator.max_questions}")
        print('='*70)
        
        # Получение вопроса
        print("🔄 Генерация вопроса...")
        question = orchestrator.get_next_question()
        
        if question.get('error'):
            print(f"❌ Ошибка генерации вопроса: {question['error']}")
            break
        
        if 'question' in question:
            print(f"\n📝 ВОПРОС: {question['question']}")
            print(f"🎯 Уровень темы: {question.get('topic_level', 'базовый')}")
            print(f"🔑 Ключевые моменты: {question.get('key_points', 'Не указаны')}")
            
            
            # Показываем информацию о Блуме (если есть)
            if question.get('bloom_level'):
                bloom_names = {
                    'remember': 'Запоминание', 'understand': 'Понимание', 'apply': 'Применение',
                    'analyze': 'Анализ', 'evaluate': 'Оценивание', 'create': 'Создание'
                }
                bloom_name = bloom_names.get(question['bloom_level'], question['bloom_level'])
                print(f"🧠 Уровень Блума: {bloom_name} ({question['bloom_level']})")
                
                if question.get('thematic_direction'):
                    print(f"🎯 Тематическое направление: {question['thematic_direction'][:100]}...")
                
                if question.get('adaptation_notes'):
                    print(f"🔧 Адаптация для студента: {question['adaptation_notes'][:100]}...")
            
            # Симуляция ответа
            if i < len(sample_answers):
                answer = sample_answers[i]
            else:
                answer = "Извините, я не знаю ответ на этот вопрос."
            
            print(f"\n👤 ОТВЕТ СТУДЕНТА: {answer}")
            
            # Оценка ответа
            print("\n🔄 Оценка ответа...")
            if not use_real_llm:
                print("   (Демонстрационный режим)")
            
            evaluation = orchestrator.submit_answer(answer)
            
            if evaluation.get('error'):
                print(f"❌ Ошибка оценки: {evaluation['error']}")
                continue
            
            print(f"📊 ОЦЕНКА: {evaluation.get('total_score', 0)}/10")
            
            # Показываем детальную оценку
            if evaluation.get('criteria_scores'):
                print(f"📈 Оценки по критериям:")
                criteria_names = {
                    'correctness': 'Правильность',
                    'completeness': 'Полнота',
                    'understanding': 'Понимание'
                }
                for criterion, score in evaluation['criteria_scores'].items():
                    name = criteria_names.get(criterion, criterion)
                    print(f"   • {name}: {score}/10")
            
            if evaluation.get('strengths'):
                print(f"✅ Сильные стороны: {evaluation['strengths']}")
            if evaluation.get('weaknesses'):
                print(f"❌ Слабые стороны: {evaluation['weaknesses']}")
            
            
            # Прогресс через LangGraph
            progress = orchestrator.get_progress()
            print(f"\n📊 ПРОГРЕСС: {progress['questions_answered']}/{progress['max_questions']} | {progress['current_score']}/{progress['max_possible_score']} баллов")
            
            # Показываем прогресс по тематической структуре
            if progress.get('theme_progress'):
                theme_progress = progress['theme_progress']
                print(f"🧠 Прогресс по Блуму: {theme_progress.get('progress_percentage', 0):.1f}%")
                current_level = theme_progress.get('current_bloom_level', 'неизвестно')
                if current_level != 'completed':
                    bloom_names = {
                        'remember': 'Запоминание', 'understand': 'Понимание', 'apply': 'Применение',
                        'analyze': 'Анализ', 'evaluate': 'Оценивание', 'create': 'Создание'
                    }
                    level_name = bloom_names.get(current_level, current_level)
                    print(f"🎯 Текущий уровень: {level_name}")
        else:
            print(f"⚠️  Проблема: {question.get('message', 'Неизвестная ошибка')}")
            break
        
        if use_real_llm:
            time.sleep(1)  # Пауза только при реальных вызовах
    
    # Финальный отчет
    print(f"\n{'='*50}")
    print("📋 РЕЗУЛЬТАТЫ ЭКЗАМЕНА")
    print('='*50)
    
    diagnostic_result = orchestrator.complete_exam()
    
    if diagnostic_result.get('error'):
        print(f"❌ Ошибка диагностики: {diagnostic_result['error']}")
        return
    
    # Показываем результаты LangGraph диагностики
    print(f"\n📋 ИТОГОВАЯ ОЦЕНКА: {diagnostic_result['grade_info']['grade'].upper()}")
    print(f"📊 Процент успеваемости: {diagnostic_result['grade_info']['percentage']}%")
    print(f"🎯 Баллы: {diagnostic_result['grade_info']['points']}")
    print(f"📝 Описание: {diagnostic_result['grade_info']['description']}")
    
    print(f"\n🔍 КРИТИЧЕСКИЕ ОБЛАСТИ:")
    for area in diagnostic_result['critical_areas']:
        print(f"   ⚠️  {area}")
    
    print(f"\n💡 ОСНОВНЫЕ РЕКОМЕНДАЦИИ:")
    for i, recommendation in enumerate(diagnostic_result['recommendations'][:5], 1):
        print(f"   {i}. {recommendation}")
    
    print(f"\n✅ Экзамен успешно завершен!")
    
    return diagnostic_result


def interactive_langgraph_exam():
    """Интерактивный режим экзамена через LangGraph оркестратор"""
    print("=== ИНТЕРАКТИВНЫЙ ЭКЗАМЕН НА LANGGRAPH ===\n")
    
    # Проверяем конфигурацию
    config = validate_yandex_config()
    if not config['is_valid']:
        print("⚠️  ВНИМАНИЕ: YandexGPT не настроен!")
        print("   Система будет работать в демонстрационном режиме.\n")
    
    # Выбор темы экзамена
    topic_manager = TopicManager()
    topic_info = topic_manager.get_topic_selection()
    
    print(f"\n📚 Выбранная тема: {topic_info['name']}")
    print(f"🎓 Предмет: {topic_info['subject']}")
    print(f"📝 Описание: {topic_info['description']}")
    print(f"⚙️  Сложность: {topic_info['difficulty']}")
    
    # Выбор режима тематической структуры
    print(f"\n🧠 Использовать тематическую структуру (Таксономия Блума)?")
    print("   Это создаст структурированную последовательность вопросов")
    print("   с автоматической адаптацией сложности")
    use_theme = input("Да/нет (по умолчанию 'да'): ").strip().lower()
    use_theme_structure = use_theme not in ['нет', 'no', 'n', 'н']
    
    # Создание LangGraph оркестратора
    print(f"\n🚀 Создание LangGraph оркестратора...")
    orchestrator = create_exam_orchestrator(
        topic_info=topic_info,
        max_questions=5,
        use_theme_structure=use_theme_structure
    )
    
    print(f"\n🎯 Экзамен готов:")
    print(f"   🧠 Тематическая структура: {'включена' if use_theme_structure else 'выключена'}")
    print("❌ Для завершения экзамена введите 'exit'\n")
    
    # Запуск экзамена
    session_info = orchestrator.start_exam("Интерактивный студент")
    print(f"🎯 Сессия экзамена: {session_info['session_id']}")
    
    # Показываем информацию о тематической структуре
    if use_theme_structure:
        theme_info = orchestrator.get_theme_structure_info()
        if not theme_info.get('error'):
            print(f"\n🧠 ТЕМАТИЧЕСКАЯ СТРУКТУРА:")
            distribution = theme_info.get('questions_distribution', {})
            bloom_names = {
                'remember': 'Запоминание', 'understand': 'Понимание', 'apply': 'Применение',
                'analyze': 'Анализ', 'evaluate': 'Оценивание', 'create': 'Создание'
            }
            
            for level, count in distribution.items():
                if count > 0:
                    name = bloom_names.get(level, level)
                    print(f"   🔹 {name}: {count} вопросов")
    
    question_count = 0
    
    while orchestrator.can_continue():
        question_count += 1
        print(f"\n{'='*40}")
        print(f"ВОПРОС {question_count} из {orchestrator.max_questions}")
        print('='*40)
        
        question = orchestrator.get_next_question()
        
        if question.get('error'):
            print(f"❌ Ошибка LangGraph: {question['error']}")
            break
        
        if 'question' not in question:
            print(f"⚠️  Проблема: {question.get('message', 'Невозможно сгенерировать вопрос')}")
            break
        
        print(f"\n📝 ВОПРОС: {question['question']}")
        print(f"🎯 Уровень: {question.get('topic_level', 'базовый')}")
        
        
        if question.get('bloom_level'):
            bloom_names = {
                'remember': 'Запоминание', 'understand': 'Понимание', 'apply': 'Применение',
                'analyze': 'Анализ', 'evaluate': 'Оценивание', 'create': 'Создание'
            }
            bloom_name = bloom_names.get(question['bloom_level'], question['bloom_level'])
            print(f"🧠 Уровень Блума: {bloom_name}")
        
        if question.get('key_points'):
            print(f"🔑 Ключевые моменты: {question['key_points']}")
        
        # Получение ответа от пользователя
        answer = input("\n👤 Ваш ответ: ").strip()
        
        if answer.lower() == 'exit':
            print("Экзамен прерван пользователем.")
            break
        
        if not answer:
            answer = "Ответ не предоставлен"
        
        # Оценка ответа
        print("\n🔄 Оценка ответа...")
        
        evaluation = orchestrator.submit_answer(answer)
        
        if evaluation.get('error'):
            print(f"❌ Ошибка оценки: {evaluation['error']}")
            continue
        
        # Показываем результат
        print(f"\n📊 ВАША ОЦЕНКА: {evaluation.get('total_score', 0)}/10")
        
        if evaluation.get('criteria_scores'):
            print(f"\n📈 Детальные оценки:")
            criteria_names = {
                'correctness': 'Правильность',
                'completeness': 'Полнота',
                'understanding': 'Понимание'
            }
            for criterion, score in evaluation['criteria_scores'].items():
                name = criteria_names.get(criterion, criterion)
                print(f"   • {name}: {score}/10")
        
        if evaluation.get('strengths'):
            print(f"\n✅ Сильные стороны: {evaluation['strengths']}")
        if evaluation.get('weaknesses'):
            print(f"❌ Слабые стороны: {evaluation['weaknesses']}")
        
        # Показать прогресс от LangGraph оркестратора
        progress = orchestrator.get_progress()
        print(f"\n📊 ОБЩИЙ ПРОГРЕСС: {progress['current_score']}/{progress['max_possible_score']} баллов ({progress['percentage']:.1f}%)")
        
        # Прогресс по тематической структуре
        if progress.get('theme_progress') and use_theme_structure:
            theme_progress = progress['theme_progress']
            print(f"🧠 Прогресс по Блуму: {theme_progress.get('progress_percentage', 0):.1f}%")
        
        # Предлагаем продолжить или завершить
        if orchestrator.can_continue():
            continue_choice = input(f"\nПродолжить экзамен? (да/нет, по умолчанию 'да'): ").strip().lower()
            if continue_choice in ['нет', 'no', 'n', 'н']:
                break
    
    # Финальная диагностика
    progress = orchestrator.get_progress()
    if progress['questions_answered'] > 0:
        print(f"\n{'='*50}")
        print("📋 РЕЗУЛЬТАТЫ ЭКЗАМЕНА")
        print('='*50)
        
        diagnostic_result = orchestrator.complete_exam()
        
        if not diagnostic_result.get('error'):
            print(f"\n🎓 ИТОГОВАЯ ОЦЕНКА: {diagnostic_result['grade_info']['grade']} ({diagnostic_result['grade_info']['percentage']}%)")
            print(f"🎯 Баллы: {diagnostic_result['grade_info']['points']}")
            
            print(f"\n💡 Основные рекомендации:")
            for i, rec in enumerate(diagnostic_result['recommendations'][:3], 1):
                print(f"   {i}. {rec}")
            
            # Предлагаем полный отчет
            full_report_choice = input(f"\nПоказать полный отчет? (да/нет): ").strip().lower()
            if full_report_choice in ['да', 'yes', 'y', 'д']:
                print(f"\n{'='*50}")
                print("📄 ПОЛНЫЙ ОТЧЕТ")
                print('='*50)
                print(diagnostic_result.get('final_report', 'Отчет недоступен'))
        else:
            print(f"❌ Ошибка диагностики: {diagnostic_result['error']}")
    else:
        print("\nЭкзамен не завершен. Данных для диагностики недостаточно.")


def demo_langgraph_agents():
    """Демонстрация работы агентов по отдельности"""
    print("=== ДЕМОНСТРАЦИЯ АГЕНТОВ ===\n")
    
    # Проверяем конфигурацию
    config = validate_yandex_config()
    if not config['is_valid']:
        print("⚠️  ВНИМАНИЕ: YandexGPT не настроен!")
        print("   Демонстрация в режиме без LLM.\n")
    
    # Выбор темы для демонстрации
    topic_manager = TopicManager()
    print("Выберите тему для демонстрации агентов:")
    topic_info = topic_manager.get_topic_selection()
    
    topic_context = topic_manager.get_topic_context_for_prompts(topic_info)
    
    print(f"\n📚 Демонстрация агентов по теме: {topic_info['name']}\n")
    
    # 1. ThemeAgent
    print("🎨 ДЕМОНСТРАЦИЯ ThemeAgent")
    print("-" * 30)
    
    try:
        theme_agent = create_theme_agent(
            subject=topic_info['subject'],
            topic_context=topic_context
        )
        
        print(f"✅ ThemeAgent создан успешно")
        print(f"📊 Информация об агенте:")
        agent_info = theme_agent.get_agent_info()
        print(f"   🆔 ID агента: {agent_info['agent_id']}")
        print(f"   📊 Операций выполнено: {agent_info['operations_count']}")
        
        if config['is_valid']:
            print("🚀 Создание тематической структуры...")
            theme_structure = theme_agent.generate_theme_structure(total_questions=5, difficulty="средний")
            
            if not theme_structure.get('error'):
                print(f"✅ Тематическая структура создана")
                print(f"📊 Всего вопросов: {theme_structure['total_questions']}")
                distribution = theme_structure.get('questions_distribution', {})
                print(f"📋 Распределение по Блуму: {len(distribution)} уровней")
            else:
                print(f"❌ Ошибка создания структуры: {theme_structure['error']}")
        else:
            print("⚠️  Демонстрация без LLM вызовов")
        
    except Exception as e:
        print(f"❌ Ошибка создания ThemeAgent: {str(e)}")
    
    # 2. QuestionAgent
    print(f"\n🤖 ДЕМОНСТРАЦИЯ QuestionAgent")
    print("-" * 30)
    
    try:
        question_agent = create_question_agent(
            subject=topic_info['subject'],
            difficulty=topic_info['difficulty'],
            topic_context=topic_context
        )
        
        print(f"✅ QuestionAgent создан успешно")
        agent_info = question_agent.get_agent_info()
        print(f"📊 ID агента: {agent_info['agent_id']}")
        
        if config['is_valid']:
            print("🚀 Генерация вопроса...")
            question_data = question_agent.generate_question(1, [])
            
            if not question_data.get('error'):
                print(f"✅ Вопрос сгенерирован: {question_data['question'][:60]}...")
                print(f"🎯 Уровень: {question_data.get('topic_level', 'неизвестно')}")
                print(f"🛡️  Приватность: {'защищена' if question_data.get('privacy_protected') else 'стандартная'}")
            else:
                print(f"❌ Ошибка генерации: {question_data['error']}")
        else:
            print("⚠️  Демонстрация без LLM вызовов")
        
    except Exception as e:
        print(f"❌ Ошибка создания QuestionAgent: {str(e)}")
    
    # 3. EvaluationAgent
    print(f"\n🔍 ДЕМОНСТРАЦИЯ EvaluationAgent")
    print("-" * 30)
    
    try:
        evaluation_agent = create_evaluation_agent(
            subject=topic_info['subject'],
            topic_context=topic_context
        )
        
        print(f"✅ EvaluationAgent создан успешно")
        agent_info = evaluation_agent.get_agent_info()
        print(f"📊 ID агента: {agent_info['agent_id']}")
        
        if config['is_valid']:
            sample_question = "Объясните основную концепцию темы"
            sample_answer = "Это важная концепция, которая включает несколько аспектов и имеет практическое применение."
            
            print(f"🚀 Оценка ответа...")
            evaluation = evaluation_agent.evaluate_answer(
                sample_question, sample_answer, "основные концепции, практическое применение", "базовый"
            )
            
            if not evaluation.get('error'):
                print(f"✅ Ответ оценен: {evaluation.get('total_score', 0)}/10")
                print(f"📊 Тип оценки: {evaluation.get('type', 'неизвестно')}")
                print(f"⏱️  Время: {evaluation.get('timestamp', 'неизвестно')}")
            else:
                print(f"❌ Ошибка оценки: {evaluation['error']}")
        else:
            print("⚠️  Демонстрация без LLM вызовов")
        
    except Exception as e:
        print(f"❌ Ошибка создания EvaluationAgent: {str(e)}")
    
    # 4. DiagnosticAgent
    print(f"\n🧠 ДЕМОНСТРАЦИЯ DiagnosticAgent")
    print("-" * 30)
    
    try:
        diagnostic_agent = create_diagnostic_agent(
            subject=topic_info['subject'],
            topic_context=topic_context
        )
        
        print(f"✅ DiagnosticAgent создан успешно")
        agent_info = diagnostic_agent.get_agent_info()
        print(f"📊 ID агента: {agent_info['agent_id']}")
        
        if config['is_valid']:
            # Симулируем данные экзамена
            mock_questions = [
                {'question': 'Тестовый вопрос 1', 'topic_level': 'базовый'},
                {'question': 'Тестовый вопрос 2', 'topic_level': 'промежуточный'}
            ]
            mock_evaluations = [
                {'total_score': 8, 'type': 'detailed', 'criteria_scores': {'correctness': 8, 'completeness': 7}},
                {'total_score': 6, 'type': 'detailed', 'criteria_scores': {'correctness': 6, 'completeness': 5}}
            ]
            
            print(f"🚀 Диагностика результатов...")
            diagnosis = diagnostic_agent.diagnose_exam_results(mock_questions, mock_evaluations)
            
            if not diagnosis.get('error'):
                print(f"✅ Диагностика завершена")
                print(f"📊 Итоговая оценка: {diagnosis['grade_info']['grade']}")
                print(f"📈 Процент: {diagnosis['grade_info']['percentage']}%")
                print(f"🔍 Критических областей: {len(diagnosis.get('critical_areas', []))}")
            else:
                print(f"❌ Ошибка диагностики: {diagnosis['error']}")
        else:
            print("⚠️  Демонстрация без LLM вызовов")
        
    except Exception as e:
        print(f"❌ Ошибка создания DiagnosticAgent: {str(e)}")
    
    print(f"\n✅ Демонстрация агентов завершена!")
    print(f"📊 Все агенты готовы к работе")


def demo_workflow_direct():
    """Демонстрация прямого использования ExamWorkflow"""
    print("=== ДЕМОНСТРАЦИЯ EXAMWORKFLOW ===\n")
    
    # Проверяем конфигурацию
    config = validate_yandex_config()
    if not config['is_valid']:
        print("⚠️  ВНИМАНИЕ: YandexGPT не настроен!")
        print("   Workflow в демонстрационном режиме.\n")
    
    # Выбор темы
    topic_manager = TopicManager()
    topic_info = topic_manager.get_topic_selection()
    
    print(f"\n📚 Создание ExamWorkflow для темы: {topic_info['name']}")
    
    # Создание workflow напрямую
    workflow = create_exam_workflow(
        topic_info=topic_info,
        max_questions=3,
        use_theme_structure=True
    )
    
    print(f"🔄 ExamWorkflow создан и готов к работе")
    
    if config['is_valid']:
        print(f"\n🚀 Запуск экзамена...")
        result = workflow.start_exam_workflow("Демо-студент")
        
        print(f"📊 РЕЗУЛЬТАТ WORKFLOW:")
        print(f"   🆔 Session: {result.get('session_id', 'неизвестно')}")
        print(f"   📊 Статус: {result.get('status', 'неизвестно')}")
        print(f"   ❓ Вопросов: {result.get('questions_count', 0)}")
        print(f"   📊 Оценок: {result.get('evaluations_count', 0)}")
        
        if result.get('errors'):
            print(f"   ❌ Ошибок: {len(result['errors'])}")
            for error in result['errors'][:3]:
                print(f"      • {error}")
        
        if result.get('diagnostic_result'):
            diagnostic = result['diagnostic_result']
            print(f"   🧠 Диагностика: {diagnostic.get('grade_info', {}).get('grade', 'неизвестно')}")
        
        if result.get('messages'):
            print(f"   📝 Сообщений workflow: {len(result['messages'])}")
    else:
        print(f"\n⚠️  Демонстрация архитектуры workflow без LLM вызовов")
        print(f"   🔄 Workflow граф готов к выполнению")
        print(f"   📊 Все состояния и переходы настроены")
    
    # Показываем статистику workflow
    workflow_stats = workflow.get_workflow_statistics()
    print(f"\n📊 СТАТИСТИКА WORKFLOW:")
    print(f"   📈 История workflow: {workflow_stats['workflow_history_count']} записей")
    print(f"   🤖 Агентов активно: {len([a for a in workflow_stats['agents_statistics'].values() if a])}")
    
    print(f"\n🏗️ АРХИТЕКТУРНЫЕ ОСОБЕННОСТИ:")
    print(f"   ✅ StateGraph управляет всем потоком экзамена")
    print(f"   ✅ Автоматические переходы между состояниями")
    print(f"   ✅ Обработка ошибок на уровне графа")
    print(f"   ✅ Централизованное управление состоянием")


def architecture_comparison():
    """Сравнение архитектур LangChain vs LangGraph"""
    print("=== СРАВНЕНИЕ АРХИТЕКТУР: LANGCHAIN VS LANGGRAPH ===\n")
    
    print("📊 УПРАВЛЕНИЕ СОСТОЯНИЕМ:")
    print("   LangChain: обычные переменные Python")
    print("   LangGraph: строгие TypedDict состояния ✅")
    
    print(f"\n🔄 АРХИТЕКТУРА:")
    print("   LangChain: простые цепочки (LLMChain)")
    print("   LangGraph: графы состояний (StateGraph) ✅")
    
    print(f"\n📊 ТИПИЗАЦИЯ:")
    print("   LangChain: слабая типизация")
    print("   LangGraph: строгая TypedDict типизация ✅")
    
    print(f"\n🔍 ОТСЛЕЖИВАЕМОСТЬ:")
    print("   LangChain: ограниченная (только логи)")
    print("   LangGraph: полная (граф выполнения) ✅")
    
    print(f"\n❌ ОБРАБОТКА ОШИБОК:")
    print("   LangChain: базовая try/catch")
    print("   LangGraph: встроенная в граф с fallback ✅")
    
    print(f"\n🛡️ ПРИВАТНОСТЬ ДАННЫХ:")
    print("   LangChain: ограниченная защита")
    print("   LangGraph: встроенная защита между агентами ✅")
    
    print(f"\n📈 ПРОИЗВОДИТЕЛЬНОСТЬ:")
    print("   LangChain: быстрые простые цепочки ✅")
    print("   LangGraph: немного медленнее из-за графов")
    
    print(f"\n🔧 РАСШИРЯЕМОСТЬ:")
    print("   LangChain: средняя (через наследование)")
    print("   LangGraph: высокая (модульные узлы) ✅")
    
    print(f"\n🎯 РЕКОМЕНДАЦИИ:")
    print("   📦 LangChain: для простых быстрых прототипов")
    print("   🚀 LangGraph: для продакшена и сложных систем ✅")
    
    print(f"\n💡 ГЛАВНЫЕ ПРЕИМУЩЕСТВА LANGGRAPH:")
    print("   ✅ Строгое управление состоянием")
    print("   ✅ Полная отслеживаемость выполнения") 
    print("   ✅ Встроенная защита приватности")
    print("   ✅ Graceful degradation при ошибках")
    print("   ✅ Легкое добавление новых узлов")
    print("   ✅ Валидация состояний на каждом этапе")


def system_status_check():
    """Проверка статуса всей LangGraph системы"""
    print("=== ПРОВЕРКА СТАТУСА LANGGRAPH СИСТЕМЫ ===\n")
    
    # 1. Проверка конфигурации
    print("🔧 КОНФИГУРАЦИЯ:")
    config = validate_yandex_config()
    print(f"   API Key: {'✅' if config['api_key'] else '❌'}")
    print(f"   Folder ID: {'✅' if config['folder_id'] else '❌'}")
    print(f"   Model ID: {config['model_id']}")
    print(f"   Система готова: {'✅' if config['is_valid'] else '❌'}")
    
    # 2. Проверка импортов
    print(f"\n📦 ИМПОРТЫ LANGGRAPH:")
    try:
        import langgraph
        print(f"   langgraph: ✅")
    except ImportError:
        print(f"   langgraph: ❌ не установлен")
    
    try:
        from langgraph.graph import StateGraph, END
        print(f"   StateGraph: ✅")
    except ImportError:
        print(f"   StateGraph: ❌ ошибка импорта")
    
    # 3. Проверка создания агентов
    print(f"\n🤖 СОЗДАНИЕ АГЕНТОВ:")
    agents_status = {}
    
    try:
        theme_agent = create_theme_agent("Тест", "Тестовый контекст")
        agents_status['ThemeAgent'] = '✅'
    except Exception as e:
        agents_status['ThemeAgent'] = f'❌ {str(e)[:50]}...'
    
    try:
        question_agent = create_question_agent("Тест", "средний", "Тестовый контекст")
        agents_status['QuestionAgent'] = '✅'
    except Exception as e:
        agents_status['QuestionAgent'] = f'❌ {str(e)[:50]}...'
    
    try:
        evaluation_agent = create_evaluation_agent("Тест", "Тестовый контекст")
        agents_status['EvaluationAgent'] = '✅'
    except Exception as e:
        agents_status['EvaluationAgent'] = f'❌ {str(e)[:50]}...'
    
    try:
        diagnostic_agent = create_diagnostic_agent("Тест", "Тестовый контекст")
        agents_status['DiagnosticAgent'] = '✅'
    except Exception as e:
        agents_status['DiagnosticAgent'] = f'❌ {str(e)[:50]}...'
    
    for agent_name, status in agents_status.items():
        print(f"   {agent_name}: {status}")
    
    # 4. Проверка workflow
    print(f"\n🔄 WORKFLOW:")
    try:
        workflow = create_exam_workflow(max_questions=1, use_theme_structure=False)
        print(f"   ExamWorkflow: ✅")
    except Exception as e:
        print(f"   ExamWorkflow: ❌ {str(e)[:50]}...")
    
    # 5. Проверка оркестратора
    print(f"\n🎼 ОРКЕСТРАТОР:")
    try:
        orchestrator = create_exam_orchestrator(max_questions=1, use_theme_structure=False)
        print(f"   ExamOrchestrator: ✅")
    except Exception as e:
        print(f"   ExamOrchestrator: ❌ {str(e)[:50]}...")
    
    # 6. Общий статус
    all_agents_ok = all('✅' in status for status in agents_status.values())
    
    print(f"\n📊 ОБЩИЙ СТАТУС:")
    if config['is_valid'] and all_agents_ok:
        print(f"   🎉 Система полностью готова к работе!")
        print(f"   🚀 Можно запускать полнофункциональные экзамены")
    elif all_agents_ok:
        print(f"   ⚠️  Система готова в демонстрационном режиме")
        print(f"   🔧 Настройте YandexGPT для полной функциональности")
    else:
        print(f"   ❌ Есть проблемы с системой")
        print(f"   🛠️  Проверьте ошибки выше")


if __name__ == "__main__":
    print("🚀 СИСТЕМА ЭКЗАМЕНИРОВАНИЯ НА LANGGRAPH")
    print("=" * 60)
    print("Выберите режим демонстрации:")
    print("1. 🎯 Основная демонстрация (рекомендуется)")
    print("2. 🤖 Интерактивный экзамен")
    print("3. 🔧 Демонстрация отдельных агентов")
    print("4. 🔄 Демонстрация ExamWorkflow")
    print("5. 📊 Сравнение архитектур")
    print("6. 🩺 Проверка статуса системы")
    
    choice = input("\nВведите номер (1-6): ").strip()
    
    try:
        if choice == "1":
            main_example()
        elif choice == "2":
            interactive_langgraph_exam()
        elif choice == "3":
            demo_langgraph_agents()
        elif choice == "4":
            demo_workflow_direct()
        elif choice == "5":
            architecture_comparison()
        elif choice == "6":
            system_status_check()
        else:
            print("Запуск основной демонстрации...")
            main_example()
    except KeyboardInterrupt:
        print("\n\n👋 Демонстрация прервана пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
        print("🩺 Запустите проверку статуса системы (опция 6)")
