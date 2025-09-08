"""
Пример использования ExamOrchestrator - единой точки входа для экзаменирования
"""
from topic_manager import TopicManager
from question_agent import QuestionAgent
from evaluation_agent import EvaluationAgent
from diagnostic_agent import DiagnosticAgent
from exam_orchestrator import ExamOrchestrator
import time


def main_example():
    """Основной пример использования специализированных агентов"""
    print("=== СИСТЕМА ЭКЗАМИНИРОВАНИЯ С 3 СПЕЦИАЛИЗИРОВАННЫМИ АГЕНТАМИ ===\n")
    
    # Выбор темы экзамена
    topic_manager = TopicManager()
    topic_info = topic_manager.get_topic_selection()
    
    print(f"\n📚 Выбранная тема: {topic_info['name']}")
    print(f"🎓 Предмет: {topic_info['subject']}")
    print(f"📝 Описание: {topic_info['description']}")
    print(f"⚙️  Сложность: {topic_info['difficulty']}")
    
    # Создание контекста для агентов
    topic_context = topic_manager.get_topic_context_for_prompts(topic_info)
    
    # Создание агентов с контекстом темы
    question_agent = QuestionAgent(
        subject=topic_info['subject'], 
        difficulty=topic_info['difficulty'],
        topic_context=topic_context
    )
    evaluation_agent = EvaluationAgent(
        subject=topic_info['subject'],
        topic_context=topic_context
    )
    diagnostic_agent = DiagnosticAgent(
        subject=topic_info['subject'],
        topic_context=topic_context
    )
    
    print("\n🤖 Агенты инициализированы:")
    print("1. QuestionAgent - для генерации умных вопросов по теме")
    print("2. EvaluationAgent - для изолированной оценки ответов")
    print("3. DiagnosticAgent - для комплексной диагностики\n")
    
    # Создание ExamOrchestrator вместо отдельных агентов
    orchestrator = ExamOrchestrator(
        topic_info=topic_info,
        max_questions=5,
        use_theme_structure=False  # Обычный режим
    )
    
    print("\n🎼 ExamOrchestrator создан и координирует:")
    print("🤖 QuestionAgent - генерация вопросов")
    print("🔍 EvaluationAgent - оценка ответов")
    print("🧠 DiagnosticAgent - финальная диагностика\n")
    
    # Симуляция экзамена через оркестратор
    simulate_orchestrated_exam(orchestrator, topic_info)


def simulate_orchestrated_exam(orchestrator: ExamOrchestrator, topic_info: dict):
    """Симуляция экзамена через ExamOrchestrator"""
    
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
    print("🚀 Запускаю экзамен через ExamOrchestrator...")
    session_info = orchestrator.start_exam("Демо-студент")
    print(f"   Сессия: {session_info['session_id']}")
    
    # Проведение экзамена
    for i in range(orchestrator.max_questions):
        print(f"\n{'='*60}")
        print(f"ВОПРОС {i + 1} ИЗ {orchestrator.max_questions}")
        print('='*60)
        
        # Получение вопроса от оркестратора
        question = orchestrator.get_next_question()
        
        if 'question' in question:
            print(f"\n📝 ВОПРОС: {question['question']}")
            print(f"🎯 Уровень: {question.get('topic_level', 'базовый')}")
            print(f"🔑 Ключевые моменты: {question.get('key_points', 'Не указаны')}")
            
            # Симуляция ответа
            if i < len(sample_answers):
                answer = sample_answers[i]
            else:
                answer = "Извините, я не знаю ответ на этот вопрос."
            
            print(f"\n👤 ОТВЕТ СТУДЕНТА: {answer}")
            
            # Оценка через оркестратор
            print("\n🔍 Оркестратор оценивает ответ...")
            evaluation = orchestrator.submit_answer(answer)
            
            print(f"📊 ОЦЕНКА: {evaluation.get('total_score', 0)}/10")
            
            if evaluation.get('strengths'):
                print(f"✅ Сильные стороны: {evaluation['strengths']}")
            if evaluation.get('weaknesses'):
                print(f"❌ Слабые стороны: {evaluation['weaknesses']}")
            
            # Прогресс
            progress = orchestrator.get_progress()
            print(f"\n📊 ПРОГРЕСС: {progress['questions_answered']}/{progress['max_questions']} | {progress['current_score']}/{progress['max_possible_score']} баллов")
            
        else:
            print(f"⚠️  Ошибка: {question.get('message', 'Неизвестная ошибка')}")
            break
        
        time.sleep(1)
    
    # Финальный отчет через оркестратор
    print(f"\n{'='*70}")
    print("🧠 Оркестратор генерирует финальный отчет...")
    print('='*70)
    
    final_report = orchestrator.get_final_report()
    if 'report' in final_report:
        print(final_report['report'])
    else:
        print("Отчет не сгенерирован или произошла ошибка")
    
    return final_report


def simulate_specialized_exam(question_agent: QuestionAgent, evaluation_agent: EvaluationAgent, diagnostic_agent: DiagnosticAgent, topic_info: dict):
    """Симулирует проведение экзамена с использованием специализированных агентов"""
    
    # Примеры ответов студента в зависимости от темы
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
    
    questions = []
    evaluations = []
    max_questions = 5
    
    for question_count in range(max_questions):
        print(f"\n{'='*60}")
        print(f"ВОПРОС {question_count + 1}")
        print('='*60)
        
        # 1. QuestionAgent генерирует вопрос с учетом предыдущих ответов
        print("🤖 QuestionAgent генерирует умный вопрос...")
        previous_answers = [{'answer': eval_data.get('answer', ''), 'score': eval_data.get('total_score', 0), 'feedback': eval_data.get('detailed_feedback', '')} for eval_data in evaluations]
        
        question_data = question_agent.generate_question(question_count + 1, previous_answers)
        questions.append(question_data)
        
        print(f"\n📝 ВОПРОС: {question_data['question']}")
        print(f"🎯 Уровень темы: {question_data['topic_level']}")
        print(f"🔑 Ключевые моменты: {question_data['key_points']}")
        
        if question_data.get('reasoning'):
            print(f"💭 Обоснование выбора: {question_data['reasoning']}")
        
        # Симуляция ответа студента
        if question_count < len(sample_answers):
            student_answer = sample_answers[question_count]
        else:
            student_answer = "Извините, я не знаю ответ на этот вопрос."
        
        print(f"\n👤 ОТВЕТ СТУДЕНТА: {student_answer}")
        
        # 2. EvaluationAgent оценивает ответ изолированно
        print("\n🔍 EvaluationAgent проводит детальную оценку...")
        evaluation_result = evaluation_agent.evaluate_answer(
            question=question_data['question'],
            student_answer=student_answer,
            key_points=question_data['key_points'],
            topic_level=question_data['topic_level'],
            detailed=True
        )
        
        # Добавляем ответ для истории
        evaluation_result['answer'] = student_answer
        evaluations.append(evaluation_result)
        
        # Показываем результат оценки
        print(f"\n📊 ОЦЕНКА: {evaluation_result['total_score']}/10")
        
        if evaluation_result['type'] == 'detailed':
            print(f"📈 Оценки по критериям:")
            for criterion, score in evaluation_result['criteria_scores'].items():
                print(f"   • {criterion}: {score}/10")
            
            print(f"\n✅ Сильные стороны: {evaluation_result['strengths']}")
            print(f"❌ Слабые стороны: {evaluation_result['weaknesses']}")
        else:
            print(f"💬 Комментарий: {evaluation_result.get('comment', '')}")
        
        # Показать прогресс
        total_score = sum(eval_data['total_score'] for eval_data in evaluations)
        max_score = len(evaluations) * 10
        print(f"\n📊 ПРОГРЕСС: {len(evaluations)}/{max_questions} вопросов | {total_score}/{max_score} баллов")
        
        # Пауза для наглядности
        time.sleep(1)
    
    # 3. DiagnosticAgent проводит финальную диагностику
    print(f"\n{'='*70}")
    print("🧠 DiagnosticAgent ПРОВОДИТ КОМПЛЕКСНУЮ ДИАГНОСТИКУ")
    print('='*70)
    
    diagnostic_result = diagnostic_agent.diagnose_exam_results(questions, evaluations)
    
    # Показываем результаты диагностики
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
    
    # Показываем полный отчет
    print(f"\n{'='*70}")
    print("📄 ПОЛНЫЙ ДИАГНОСТИЧЕСКИЙ ОТЧЕТ")
    print('='*70)
    print(diagnostic_result['final_report'])
    
    # Дорожная карта обучения
    roadmap = diagnostic_agent.generate_learning_roadmap(diagnostic_result)
    print(f"\n{'='*70}")
    print("🗺️  ДОРОЖНАЯ КАРТА ОБУЧЕНИЯ")
    print('='*70)
    
    print("🚨 НЕМЕДЛЕННЫЕ ДЕЙСТВИЯ:")
    for action in roadmap['immediate_actions']:
        print(f"   • {action}")
    
    print("\n📅 КРАТКОСРОЧНЫЕ ЦЕЛИ:")
    for goal in roadmap['short_term_goals']:
        print(f"   • {goal}")
    
    if roadmap['medium_term_goals']:
        print("\n📆 СРЕДНЕСРОЧНЫЕ ЦЕЛИ:")
        for goal in roadmap['medium_term_goals']:
            print(f"   • {goal}")
    
    if roadmap['long_term_goals']:
        print("\n🎯 ДОЛГОСРОЧНЫЕ ЦЕЛИ:")
        for goal in roadmap['long_term_goals']:
            print(f"   • {goal}")
    
    return diagnostic_result


def interactive_exam():
    """Интерактивный режим экзамена через ExamOrchestrator"""
    print("=== ИНТЕРАКТИВНЫЙ ЭКЗАМЕН ЧЕРЕЗ EXAMORCHESTRATOR ===\n")
    
    # Выбор темы экзамена
    topic_manager = TopicManager()
    topic_info = topic_manager.get_topic_selection()
    
    print(f"\n📚 Выбранная тема: {topic_info['name']}")
    print(f"🎓 Предмет: {topic_info['subject']}")
    print(f"📝 Описание: {topic_info['description']}")
    print(f"⚙️  Сложность: {topic_info['difficulty']}")
    
    if topic_info.get('key_concepts'):
        print(f"🔑 Ключевые концепции: {', '.join(topic_info['key_concepts'][:3])}...")
    
    # Создание ExamOrchestrator
    orchestrator = ExamOrchestrator(
        topic_info=topic_info,
        max_questions=5,
        use_theme_structure=False
    )
    
    print(f"\n🚀 Начинаем интерактивный экзамен по теме '{topic_info['name']}'")
    print("🎼 Оркестратор координирует все аспекты экзамена!")
    print("❌ Для завершения экзамена введите 'exit'\n")
    
    # Запуск экзамена
    session_info = orchestrator.start_exam("Интерактивный студент")
    print(f"🎯 Сессия экзамена: {session_info['session_id']}")
    
    question_count = 0
    
    while orchestrator.can_continue():
        question_count += 1
        print(f"\n{'='*50}")
        print(f"ВОПРОС {question_count} из {orchestrator.max_questions}")
        print('='*50)
        
        # Получение вопроса от оркестратора
        print("🎼 Оркестратор генерирует персонализированный вопрос...")
        
        question = orchestrator.get_next_question()
        
        if 'question' not in question:
            print(f"⚠️  Ошибка: {question.get('message', 'Невозможно сгенерировать вопрос')}")
            break
        
        print(f"\n📝 ВОПРОС: {question['question']}")
        print(f"🎯 Уровень: {question.get('topic_level', 'базовый')}")
        
        if question.get('reasoning'):
            print(f"💭 Почему этот вопрос: {question['reasoning']}")
        if question.get('key_points'):
            print(f"🔑 Ключевые моменты: {question['key_points']}")
        
        # Получение ответа от пользователя
        answer = input("\n👤 Ваш ответ: ").strip()
        
        if answer.lower() == 'exit':
            print("Экзамен прерван пользователем.")
            break
        
        if not answer:
            answer = "Ответ не предоставлен"
        
        # Оценка ответа через оркестратор
        print("\n🎼 Оркестратор оценивает ответ...")
        evaluation = orchestrator.submit_answer(answer)
        
        # Показываем результат
        print(f"\n📊 ВАША ОЦЕНКА: {evaluation.get('total_score', 0)}/10")
        
        if evaluation.get('criteria_scores'):
            print(f"\n📈 Оценки по критериям:")
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
        
        # Показать прогресс от оркестратора
        progress = orchestrator.get_progress()
        print(f"\n📊 ОБЩИЙ ПРОГРЕСС: {progress['current_score']}/{progress['max_possible_score']} баллов ({progress['percentage']:.1f}%)")
        
        # Предлагаем продолжить или завершить
        if orchestrator.can_continue():
            continue_choice = input(f"\nПродолжить экзамен? (да/нет, по умолчанию 'да'): ").strip().lower()
            if continue_choice in ['нет', 'no', 'n', 'н']:
                orchestrator.force_complete()
                break
    
    # Финальная диагностика через оркестратор
    progress = orchestrator.get_progress()
    if progress['questions_answered'] > 0:
        print(f"\n{'='*60}")
        print("🎼 Оркестратор генерирует финальный отчет...")
        print('='*60)
        
        final_report = orchestrator.get_final_report()
        
        if 'error' not in final_report:
            print(f"\n🎓 ФИНАЛЬНЫЙ ОТЧЕТ:")
            if 'report' in final_report:
                report_lines = final_report['report'].split('\n')[:10]  # Первые 10 строк
                for line in report_lines:
                    if line.strip():
                        print(f"   {line}")
                if len(final_report['report'].split('\n')) > 10:
                    print("   ... (полный отчет доступен по запросу)")
            
            # Предлагаем полный отчет
            full_report_choice = input(f"\nПоказать полный отчет? (да/нет): ").strip().lower()
            if full_report_choice in ['да', 'yes', 'y', 'д'] and 'report' in final_report:
                print(f"\n{'='*70}")
                print("📄 ПОЛНЫЙ ОТЧЕТ ОТ EXAMORCHESTRATOR")
                print('='*70)
                print(final_report['report'])
        else:
            print(f"⚠️  Ошибка при генерации отчета: {final_report.get('error', 'Неизвестная ошибка')}")
    
    else:
        print("\nЭкзамен не завершен. Данных для отчета недостаточно.")


def demo_individual_agents():
    """Демонстрация работы каждого агента по отдельности"""
    print("=== ДЕМОНСТРАЦИЯ РАБОТЫ ОТДЕЛЬНЫХ АГЕНТОВ ===\n")
    
    # Выбор темы для демонстрации
    topic_manager = TopicManager()
    print("Выберите тему для демонстрации агентов:")
    topic_info = topic_manager.get_topic_selection()
    
    topic_context = topic_manager.get_topic_context_for_prompts(topic_info)
    
    print(f"\n📚 Демонстрация будет проведена по теме: {topic_info['name']}\n")
    
    # 1. QuestionAgent
    print("🤖 ДЕМОНСТРАЦИЯ QuestionAgent")
    print("-" * 40)
    question_agent = QuestionAgent(
        subject=topic_info['subject'], 
        difficulty=topic_info['difficulty'],
        topic_context=topic_context
    )
    
    # Первый вопрос
    print("Генерирую первый вопрос по теме...")
    q1 = question_agent.generate_question(1)
    print(f"Вопрос 1: {q1['question']}")
    print(f"Уровень: {q1['topic_level']}")
    print(f"Ключевые моменты: {q1['key_points']}")
    
    # Симулируем слабый ответ
    weak_answers = [{'answer': 'Не знаю', 'score': 3, 'feedback': 'Слабое понимание темы'}]
    print(f"\nГенерирую второй вопрос с учетом слабого первого ответа...")
    q2 = question_agent.generate_question(2, weak_answers)
    print(f"Вопрос 2: {q2['question']}")
    print(f"Обоснование: {q2.get('reasoning', 'Не указано')}")
    
    # 2. EvaluationAgent
    print(f"\n🔍 ДЕМОНСТРАЦИЯ EvaluationAgent")
    print("-" * 40)
    evaluation_agent = EvaluationAgent(
        subject=topic_info['subject'],
        topic_context=topic_context
    )
    
    # Используем вопрос по выбранной теме
    if "python" in topic_info['name'].lower():
        sample_question = "Объясните разницу между циклами for и while в Python"
        sample_answer = "Цикл for используется для итерации по последовательности элементов, а while выполняется пока условие истинно. For удобнее когда знаем количество итераций."
        key_points = "итерация по последовательности, условие выполнения, использование range, контроль итераций"
    elif "фотоэффект" in topic_info['name'].lower():
        sample_question = "Сформулируйте уравнение Эйнштейна для фотоэффекта"
        sample_answer = "Уравнение Эйнштейна: E = hν = A + Ek, где E - энергия фотона, A - работа выхода, Ek - кинетическая энергия фотоэлектрона."
        key_points = "энергия фотона, работа выхода, кинетическая энергия, квантовая природа света"
    else:
        sample_question = q1['question']
        sample_answer = "Это сложная тема, требующая детального изучения основных принципов и их практического применения."
        key_points = q1['key_points']
    
    print(f"Оцениваю ответ: '{sample_answer[:50]}...'")
    evaluation = evaluation_agent.evaluate_answer(sample_question, sample_answer, key_points, "базовый")
    
    print(f"Итоговая оценка: {evaluation['total_score']}/10")
    if evaluation['type'] == 'detailed':
        print("Оценки по критериям:")
        for criterion, score in evaluation['criteria_scores'].items():
            print(f"  • {criterion}: {score}/10")
    
    # 3. DiagnosticAgent
    print(f"\n🧠 ДЕМОНСТРАЦИЯ DiagnosticAgent")
    print("-" * 40)
    diagnostic_agent = DiagnosticAgent(
        subject=topic_info['subject'],
        topic_context=topic_context
    )
    
    # Симулируем данные экзамена по теме
    mock_questions = [q1, q2]
    mock_evaluations = [
        {'total_score': 8, 'type': 'detailed', 'criteria_scores': {'correctness': 8, 'completeness': 7}, 'answer': 'Хороший ответ'},
        {'total_score': 6, 'type': 'detailed', 'criteria_scores': {'correctness': 6, 'completeness': 5}, 'answer': 'Частично правильный ответ'}
    ]
    
    print("Провожу диагностику результатов по теме...")
    diagnosis = diagnostic_agent.diagnose_exam_results(mock_questions, mock_evaluations)
    
    print(f"Итоговая оценка: {diagnosis['grade_info']['grade']}")
    print(f"Процент: {diagnosis['grade_info']['percentage']}%")
    print(f"Критические области: {', '.join(diagnosis['critical_areas'][:2])}")
    
    print("\n✅ Демонстрация отдельных агентов завершена!")
    print(f"Все агенты настроены для работы с темой: {topic_info['name']}")


def demo_individual_agents():
    """Демонстрация работы ExamOrchestrator в разных режимах"""
    print("=== ДЕМОНСТРАЦИЯ EXAMORCHESTRATOR В РАЗНЫХ РЕЖИМАХ ===\n")
    
    # Выбор темы для демонстрации
    topic_manager = TopicManager()
    print("Выберите тему для демонстрации:")
    topic_info = topic_manager.get_topic_selection()
    
    print(f"\n📚 Демонстрация будет проведена по теме: {topic_info['name']}\n")
    
    # 1. Обычный режим ExamOrchestrator
    print("🎼 ДЕМОНСТРАЦИЯ 1: Обычный режим ExamOrchestrator")
    print("-" * 60)
    
    orchestrator1 = ExamOrchestrator(
        topic_info=topic_info,
        max_questions=3,
        use_theme_structure=False
    )
    
    session1 = orchestrator1.start_exam("Демо-студент 1")
    print(f"Сессия: {session1['session_id']}")
    
    # Генерируем и оцениваем один вопрос
    q1 = orchestrator1.get_next_question()
    if 'question' in q1:
        print(f"Вопрос: {q1['question']}")
        print(f"Уровень: {q1.get('topic_level', 'базовый')}")
        
        # Симуляция ответа
        demo_answer = "Это основной концепт в данной области, который требует понимания."
        print(f"Ответ: {demo_answer}")
        
        evaluation = orchestrator1.submit_answer(demo_answer)
        print(f"Оценка: {evaluation.get('total_score', 0)}/10")
    
    # 2. Тематический режим
    print(f"\n🧠 ДЕМОНСТРАЦИЯ 2: Тематический режим (Блум)")
    print("-" * 60)
    
    orchestrator2 = ExamOrchestrator(
        topic_info=topic_info,
        max_questions=3,
        use_theme_structure=True
    )
    
    theme_info = orchestrator2.get_theme_structure_info()
    print(f"Тематическая структура создана: {theme_info.get('total_questions', 0)} вопросов")
    
    if theme_info.get('questions_distribution'):
        bloom_names = {
            'remember': 'Запоминание', 'understand': 'Понимание', 
            'apply': 'Применение', 'analyze': 'Анализ'
        }
        print("Распределение по уровням Блума:")
        for level, count in theme_info['questions_distribution'].items():
            name = bloom_names.get(level, level)
            print(f"  • {name}: {count} вопросов")
    
    # 3. Сравнение режимов
    print(f"\n🔄 СРАВНЕНИЕ РЕЖИМОВ:")
    print("🎼 Обычный ExamOrchestrator: гибкая генерация вопросов")
    print("🧠 Тематический: структурированный подход по Блуму")
    print("✅ Оба режима используют единую архитектуру ExamOrchestrator")
    
    print(f"\n✅ Демонстрация завершена!")
    print(f"ExamOrchestrator обеспечивает координацию всех агентов в обоих режимах!")


def quick_topic_demo():
    """Быстрая демонстрация готовых тем"""
    print("=== БЫСТРАЯ ДЕМОНСТРАЦИЯ ГОТОВЫХ ТЕМ ===\n")
    
    topic_manager = TopicManager()
    topics = topic_manager.get_predefined_topics()
    
    print("📚 Доступные готовые темы:")
    for i, (key, topic) in enumerate(topics.items(), 1):
        print(f"{i}. {topic['name']} ({topic['subject']})")
        print(f"   📝 {topic['description']}")
        print(f"   🔑 Ключевые концепции: {', '.join(topic['key_concepts'][:3])}...")
        print()
    
    print("💡 Каждая тема содержит:")
    print("   • Специализированные вопросы по теме")
    print("   • Контекстную оценку ответов")
    print("   • Персональные рекомендации")
    print()
    
    choice = input("Хотите попробовать интерактивный экзамен по одной из тем? (да/нет): ").strip().lower()
    if choice in ['да', 'yes', 'y', 'д']:
        interactive_exam()


def theme_structure_demo():
    """Демонстрация экзамена с тематической структурой"""
    print("=== ЭКЗАМЕН С ТЕМАТИЧЕСКОЙ СТРУКТУРОЙ ===\n")
    
    # Выбор темы
    topic_manager = TopicManager()
    topic_info = topic_manager.get_topic_selection()
    
    print(f"\n📚 Создаю тематическую структуру для темы: {topic_info['name']}")
    print("🔄 ThemeAgent создает руководящие принципы, QuestionAgent генерирует вопросы")
    
    # Создание оркестратора с тематической структурой
    orchestrator = ExamOrchestrator(
        topic_info=topic_info,
        max_questions=8,
        use_theme_structure=True
    )
    
    # Получение информации о структуре
    theme_info = orchestrator.get_theme_structure_info()
    print(f"\n📊 ТЕМАТИЧЕСКАЯ СТРУКТУРА:")
    print(f"   📝 Всего вопросов: {theme_info['total_questions']}")
    print(f"   ⏱️  Время экзамена: {theme_info['estimated_duration']} минут")
    print(f"   🧠 ID структуры: {theme_info['curriculum_id']}")
    
    print(f"\n📋 Распределение по уровням Блума:")
    distribution = theme_info['questions_distribution']
    bloom_names = {
        'remember': 'Запоминание', 'understand': 'Понимание', 'apply': 'Применение',
        'analyze': 'Анализ', 'evaluate': 'Оценивание', 'create': 'Создание'
    }
    
    for level, count in distribution.items():
        name = bloom_names.get(level, level)
        print(f"   🔹 {name}: {count} вопросов")
    
    print(f"\n💡 РУКОВОДЯЩИЕ ПРИНЦИПЫ:")
    guidelines = theme_info.get('question_guidelines', {})
    for level, guideline_info in guidelines.items():
        level_name = guideline_info.get('level_name', level)
        print(f"   📖 {level_name}: принципы созданы для {guideline_info.get('question_count', 0)} вопросов")
    
    # Запуск экзамена
    print(f"\n🚀 Запускаю экзамен...")
    session_info = orchestrator.start_exam("Студент по структуре")
    print(f"   Сессия: {session_info['session_id']}")
    
    # Демонстрация нескольких вопросов
    for i in range(min(4, theme_info['total_questions'])):
        print(f"\n{'='*50}")
        print(f"ВОПРОС {i+1} (СГЕНЕРИРОВАН QuestionAgent ПО ПРИНЦИПАМ ThemeAgent)")
        print('='*50)
        
        # Получение вопроса
        question = orchestrator.get_next_question()
        
        if 'question' in question:
            bloom_level = question.get('bloom_level', 'unknown')
            bloom_name = bloom_names.get(bloom_level, bloom_level)
            
            print(f"🧠 Уровень Блума: {bloom_name} ({bloom_level})")
            print(f"📊 Уровень темы: {question.get('topic_level', 'базовый')}")
            print(f"❓ Вопрос: {question['question']}")
            
            # Показать тематическое направление
            if 'thematic_direction' in question:
                print(f"🎯 Направление: {question['thematic_direction'][:100]}...")
            
            # Показать адаптацию
            if 'adaptation_notes' in question:
                print(f"🔧 Адаптация: {question['adaptation_notes'][:100]}...")
            
            # Показать защиту приватности
            if question.get('privacy_protected'):
                print(f"🛡️ Приватность: защищена (QuestionAgent не видел тексты ответов)")
                print(f"📊 Данные: {question.get('evaluation_summaries_count', 0)} характеристик оценок")
            
            # Демонстрационный ответ
            demo_answers = {
                'remember': "Это базовое определение или факт по теме",
                'understand': "Это объяснение концепции с демонстрацией понимания", 
                'apply': "Это практическое применение знаний в конкретной ситуации",
                'analyze': "Это детальный анализ компонентов и связей",
                'evaluate': "Это критическая оценка с обоснованием критериев",
                'create': "Это создание нового решения или синтез идей"
            }
            
            demo_answer = demo_answers.get(bloom_level, "Демонстрационный ответ студента")
            print(f"\n👤 ДЕМО-ОТВЕТ: {demo_answer}")
            
            # Оценка
            evaluation = orchestrator.submit_answer(demo_answer)
            print(f"📊 ОЦЕНКА: {evaluation.get('total_score', 0)}/10")
            
            # Прогресс по структуре
            progress = orchestrator.get_theme_progress_detailed()
            print(f"\n📈 Прогресс: {progress['progress_percentage']:.1f}%")
            print(f"🎯 Текущий уровень: {bloom_names.get(progress['current_bloom_level'], 'завершено')}")
            
        else:
            print(f"⚠️  Ошибка: {question.get('error', 'Неизвестная ошибка')}")
    
    # Итоговый отчет
    print(f"\n{'='*60}")
    print("📄 ИТОГОВЫЙ ОТЧЕТ ПО ТЕМАТИЧЕСКОЙ СТРУКТУРЕ")
    print('='*60)
    
    theme_report = orchestrator.get_theme_summary_report()
    print(theme_report)
    
    # Валидация структуры
    validation = orchestrator.validate_theme_structure()
    if not validation.get('error'):
        print(f"\n✅ ВАЛИДАЦИЯ СТРУКТУРЫ:")
        print(f"   Валидна: {'Да' if validation['is_valid'] else 'Нет'}")
        if validation['warnings']:
            print(f"   ⚠️  Предупреждения: {len(validation['warnings'])}")
        if validation['recommendations']:
            print(f"   💡 Рекомендации: {len(validation['recommendations'])}")
    
    print(f"\n🔄 АРХИТЕКТУРА:")
    print("   1. ThemeAgent создал тематическую структуру и принципы")
    print("   2. QuestionAgent сгенерировал вопросы на основе принципов")
    print("   3. Каждый вопрос уникален и соответствует требованиям структуры")
    
    print(f"\n🛡️ ПРИВАТНОСТЬ:")
    print("   ✅ QuestionAgent НЕ видит тексты ответов студента")
    print("   ✅ Передаются только характеристики оценок от EvaluationAgent")
    print("   ✅ Адаптация происходит на основе метрик, а не содержания")
    print("   ✅ Полная приватность ответов студента сохраняется")


if __name__ == "__main__":
    print("🎓 СИСТЕМА ЭКЗАМЕНИРОВАНИЯ С ТЕМАТИЧЕСКОЙ СТРУКТУРОЙ")
    print("=" * 55)
    print("Выберите режим:")
    print("1. Демонстрационный экзамен (с выбором темы)")
    print("2. Интерактивный экзамен")
    print("3. Демонстрация отдельных агентов")
    print("4. Обзор готовых тем")
    print("5. 🧠 ЭКЗАМЕН С ТЕМАТИЧЕСКОЙ СТРУКТУРОЙ (ThemeAgent + QuestionAgent)")
    
    choice = input("\nВведите номер (1-5): ").strip()
    
    if choice == "2":
        interactive_exam()
    elif choice == "3":
        demo_individual_agents()
    elif choice == "4":
        quick_topic_demo()
    elif choice == "5":
        theme_structure_demo()
    else:
        main_example()