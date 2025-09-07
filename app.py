"""
Streamlit приложение-чатбот для интеллектуальной системы диагностики
"""
import streamlit as st
import sys
import os
import json
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Добавляем путь к модулям агентов
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

from exam_orchestrator import ExamOrchestrator
from topic_manager import TopicManager

# Конфигурация страницы
st.set_page_config(
    page_title="🎓 Скользящая диагностика",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Никаких пользовательских стилей - используем дефолтный Streamlit

def initialize_session_state():
    """Инициализация состояния сессии"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    
    if 'exam_started' not in st.session_state:
        st.session_state.exam_started = False
    
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    
    if 'waiting_for_answer' not in st.session_state:
        st.session_state.waiting_for_answer = False
    
    if 'exam_completed' not in st.session_state:
        st.session_state.exam_completed = False
    
    if 'topic_selected' not in st.session_state:
        st.session_state.topic_selected = False
    
    if 'student_name' not in st.session_state:
        st.session_state.student_name = "Студент"
    
    if 'first_question_generated' not in st.session_state:
        st.session_state.first_question_generated = False

def add_message(role, content, message_type="text"):
    """Добавление сообщения в чат"""
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "type": message_type,
        "timestamp": datetime.now()
    })

def display_chat_messages():
    """Отображение истории чата"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.write("**👤 Ваш ответ:**")
            st.write(message['content'])
            st.divider()
        
        elif message["role"] == "assistant":
            if message.get("type") == "question":
                st.write("**❓ Вопрос:**")
                st.write(message['content'])
                st.divider()
            
            elif message.get("type") == "evaluation":
                st.write("**📊 Оценка:**")
                st.write(message['content'])
                st.divider()
            
            else:
                st.info(message['content'])
                st.divider()

def setup_exam():
    """Настройка экзамена"""
    st.sidebar.header("⚙️ Настройки экзамена")
    
    # Имя студента
    student_name = st.sidebar.text_input("Имя студента", value=st.session_state.student_name)
    if student_name != st.session_state.student_name:
        st.session_state.student_name = student_name
    
    # Выбор способа задания темы
    topic_source = st.sidebar.radio(
        "Способ выбора темы:",
        options=["Готовые темы", "Своя тема"],
        index=0
    )
    
    topic_manager = TopicManager()
    
    if topic_source == "Готовые темы":
        # Выбор из предопределенных тем
        topics = topic_manager.get_predefined_topics()
        
        topic_options = {f"{topic['name']} ({topic['subject']})": key 
                        for key, topic in topics.items()}
        
        selected_topic_display = st.sidebar.selectbox(
            "Выберите тему экзамена",
            options=list(topic_options.keys()),
            index=0
        )
        
        selected_topic_key = topic_options[selected_topic_display]
        raw_topic = topics[selected_topic_key]
        
        # Выбор уровня сложности
        difficulty_options = ["легкий", "средний", "сложный"]
        selected_difficulty = st.sidebar.selectbox(
            "Уровень сложности",
            options=difficulty_options,
            index=1  # средний по умолчанию
        )
        
        # Создаем правильную структуру topic_info с полем difficulty
        topic_info = {
            'type': 'predefined',
            'key': selected_topic_key,
            'name': raw_topic['name'],
            'subject': raw_topic['subject'],
            'description': raw_topic['description'],
            'difficulty': selected_difficulty,
            'key_concepts': raw_topic['key_concepts']
        }
    
    else:
        # Ввод пользовательской темы
        custom_name = st.sidebar.text_input(
            "Название темы:",
            placeholder="Например: Квантовая физика"
        )
        
        custom_subject = st.sidebar.text_input(
            "Предмет:",
            placeholder="Например: Физика",
            value="Общие знания"
        )
        
        custom_description = st.sidebar.text_area(
            "Описание темы (необязательно):",
            placeholder="Краткое описание того, что будет изучаться...",
            height=80
        )
        
        # Выбор уровня сложности
        difficulty_options = ["легкий", "средний", "сложный"]
        selected_difficulty = st.sidebar.selectbox(
            "Уровень сложности",
            options=difficulty_options,
            index=1  # средний по умолчанию
        )
        
        # Ключевые концепции (необязательно)
        custom_concepts_input = st.sidebar.text_area(
            "Ключевые концепции (через запятую):",
            placeholder="концепция1, концепция2, концепция3...",
            height=60
        )
        
        # Обработка ключевых концепций
        key_concepts = []
        if custom_concepts_input.strip():
            key_concepts = [concept.strip() for concept in custom_concepts_input.split(',') if concept.strip()]
        
        # Создаем структуру для пользовательской темы
        topic_info = {
            'type': 'custom',
            'key': 'custom',
            'name': custom_name if custom_name.strip() else "Пользовательская тема",
            'subject': custom_subject if custom_subject.strip() else "Общие знания",
            'description': custom_description if custom_description.strip() else f"Экзамен по теме: {custom_name if custom_name.strip() else 'Пользовательская тема'}",
            'difficulty': selected_difficulty,
            'key_concepts': key_concepts
        }
    
    # Настройки экзамена
    max_questions = st.sidebar.slider("Количество вопросов", 3, 10, 5)
    use_theme_structure = st.sidebar.checkbox("Использовать структуру по Блуму", False)  # Отключено по умолчанию для скорости
    
    # Проверка готовности к началу экзамена
    can_start_exam = True
    if topic_source == "Своя тема":
        if not custom_name.strip():
            can_start_exam = False
            st.sidebar.warning("⚠️ Введите название темы")
    
    # Кнопка начала экзамена
    exam_disabled = st.session_state.exam_started or not can_start_exam
    if st.sidebar.button("🚀 Начать экзамен", disabled=exam_disabled):
        start_exam(topic_info, max_questions, use_theme_structure)
    
    # Информация о теме
    if not st.session_state.exam_started:
        st.sidebar.markdown("### 📚 Информация о теме")
        st.sidebar.write(f"**Предмет:** {topic_info['subject']}")
        st.sidebar.write(f"**Описание:** {topic_info['description']}")
        st.sidebar.write(f"**Уровень сложности:** {topic_info['difficulty']}")
        
        if topic_info.get('key_concepts'):
            st.sidebar.write("**Ключевые концепции:**")
            for concept in topic_info['key_concepts'][:5]:
                st.sidebar.write(f"• {concept}")

def start_exam(topic_info, max_questions, use_theme_structure):
    """Запуск экзамена"""
    # Показываем индикатор загрузки
    with st.spinner("Инициализация экзамена..."):
        try:
            # Создание оркестратора
            st.session_state.orchestrator = ExamOrchestrator(
                topic_info=topic_info,
                max_questions=max_questions,
                use_theme_structure=use_theme_structure
            )
            
            # Запуск экзамена
            session_info = st.session_state.orchestrator.start_exam(st.session_state.student_name)
            
            st.session_state.exam_started = True
            st.session_state.topic_selected = True
            st.session_state.exam_completed = False
            st.session_state.first_question_generated = False
            
            # Приветственное сообщение
            welcome_msg = f"""Добро пожаловать, {st.session_state.student_name}! 🎓

**Тема:** {topic_info['name']}
**Предмет:** {topic_info['subject']}
**Вопросов:** {max_questions}
**Режим:** {'Структурированный' if use_theme_structure else 'Быстрый'}

Экзамен готов к началу! Нажмите кнопку ниже для получения первого вопроса."""
            
            add_message("assistant", welcome_msg)
            
        except Exception as e:
            st.error(f"Ошибка при запуске экзамена: {str(e)}")
            return
    
    st.rerun()

def get_next_question():
    """Получение следующего вопроса"""
    if not st.session_state.orchestrator:
        return
    
    try:
        question_data = st.session_state.orchestrator.get_next_question()
        
        if 'question' in question_data:
            st.session_state.current_question = question_data
            st.session_state.waiting_for_answer = True
            
            # Упрощенное форматирование вопроса
            question_text = f"""**Вопрос {question_data.get('question_number', '?')}**

{question_data['question']}

*Уровень: {question_data.get('topic_level', 'базовый')}*"""
            
            add_message("assistant", question_text, "question")
            
        elif 'message' in question_data:
            # Экзамен завершен
            st.session_state.exam_completed = True
            st.session_state.waiting_for_answer = False
            add_message("assistant", f"📝 {question_data['message']}")
            generate_final_report()
            
        else:
            add_message("assistant", f"❌ Ошибка: {question_data.get('error', 'Неизвестная ошибка')}")
            
    except Exception as e:
        st.error(f"Ошибка при получении вопроса: {str(e)}")

def submit_answer(answer):
    """Отправка ответа на оценку"""
    if not st.session_state.orchestrator or not st.session_state.current_question:
        return
    
    # Добавляем ответ пользователя в чат
    add_message("user", answer)
    
    # Показываем индикатор загрузки для оценки
    with st.spinner("Оценка вашего ответа..."):
        try:
            # Отправляем ответ на оценку
            evaluation = st.session_state.orchestrator.submit_answer(answer)
            
            # Упрощенное форматирование результата оценки
            eval_text = f"""**Оценка: {evaluation.get('total_score', 0)}/10 баллов**"""
            
            if evaluation.get('criteria_scores'):
                eval_text += "\n\n**Оценки по критериям:**\n"
                criteria_names = {
                    'correctness': 'Правильность',
                    'completeness': 'Полнота', 
                    'understanding': 'Понимание'
                }
                for criterion, score in evaluation['criteria_scores'].items():
                    name = criteria_names.get(criterion, criterion)
                    eval_text += f"• {name}: {score}/10\n"
            
            if evaluation.get('strengths'):
                eval_text += f"\n**✅ Сильные стороны:** {evaluation['strengths']}"
            
            if evaluation.get('weaknesses'):
                eval_text += f"\n\n**❌ Области для улучшения:** {evaluation['weaknesses']}"
            
            add_message("assistant", eval_text, "evaluation")
            
            st.session_state.waiting_for_answer = False
            st.session_state.current_question = None
            
        except Exception as e:
            st.error(f"Ошибка при обработке ответа: {str(e)}")
            return
    
    # Проверяем, можно ли продолжить
    if st.session_state.orchestrator.can_continue():
        with st.spinner("Подготовка следующего вопроса..."):
            time.sleep(0.5)
            get_next_question()
    else:
        # Экзамен завершен
        st.session_state.exam_completed = True
        generate_final_report()
    
    st.rerun()

def generate_final_report():
    """Генерация финального отчета"""
    if not st.session_state.orchestrator:
        return
    
    try:
        # Завершаем экзамен если еще не завершен
        if st.session_state.orchestrator.exam_session['status'] == 'in_progress':
            st.session_state.orchestrator.force_complete()
        
        # Получаем финальный отчет через complete_exam
        final_report = st.session_state.orchestrator.complete_exam()
        
        if 'error' not in final_report:
            report_text = f"""🎉 **Экзамен завершен!**

**Итоговые результаты:**
• Оценка: {final_report['grade_info']['grade'].upper()}
• Успеваемость: {final_report['grade_info']['percentage']}%
• Баллы: {final_report['grade_info']['points']}

**Описание:** {final_report['grade_info']['description']}

**Рекомендации:**"""
            
            for i, recommendation in enumerate(final_report['recommendations'][:3], 1):
                report_text += f"\n{i}. {recommendation}"
            
            add_message("assistant", report_text)
        else:
            add_message("assistant", f"❌ Ошибка при генерации отчета: {final_report.get('error', 'Неизвестная ошибка')}")
    
    except Exception as e:
        st.error(f"Ошибка при генерации финального отчета: {str(e)}")

def display_progress():
    """Отображение прогресса экзамена"""
    if not st.session_state.orchestrator or not st.session_state.exam_started:
        return
    
    try:
        progress = st.session_state.orchestrator.get_progress()
        
        st.sidebar.markdown("### 📊 Прогресс экзамена")
        
        # Прогресс-бар
        progress_percentage = (progress['questions_answered'] / progress['max_questions']) * 100
        st.sidebar.progress(progress_percentage / 100)
        
        # Метрики
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Вопросы", f"{progress['questions_answered']}/{progress['max_questions']}")
        with col2:
            st.metric("Баллы", f"{progress['current_score']}/{progress['max_possible_score']}")
        
        # Процент успеваемости
        if progress['max_possible_score'] > 0:
            success_rate = (progress['current_score'] / progress['max_possible_score']) * 100
            st.sidebar.metric("Успеваемость", f"{success_rate:.1f}%")
        
        # Статус
        status_emoji = {
            'in_progress': '🟡',
            'completed': '🟢',
            'not_started': '🔴'
        }
        st.sidebar.write(f"**Статус:** {status_emoji.get(progress['status'], '❓')} {progress['status']}")
        
    except Exception as e:
        st.sidebar.error(f"Ошибка при отображении прогресса: {str(e)}")

def display_analytics():
    """Отображение аналитики в боковой панели"""
    if not st.session_state.orchestrator or not st.session_state.exam_started:
        return
    
    try:
        progress = st.session_state.orchestrator.get_progress()
        
        if progress['questions_answered'] > 0:
            st.sidebar.markdown("### 📈 Аналитика")
            
            # График баллов по вопросам
            evaluations = st.session_state.orchestrator.exam_session['evaluations']
            if evaluations:
                scores = [eval_data.get('total_score', 0) for eval_data in evaluations]
                question_numbers = list(range(1, len(scores) + 1))
                
                fig = px.line(
                    x=question_numbers, 
                    y=scores,
                    title="Баллы по вопросам",
                    labels={'x': 'Номер вопроса', 'y': 'Баллы'}
                )
                fig.update_layout(height=300, showlegend=False)
                st.sidebar.plotly_chart(fig, use_container_width=True)
                
                # Средний балл
                avg_score = sum(scores) / len(scores)
                st.sidebar.metric("Средний балл", f"{avg_score:.1f}/10")
    
    except Exception as e:
        st.sidebar.error(f"Ошибка при отображении аналитики: {str(e)}")

def main():
    """Основная функция приложения"""
    initialize_session_state()
    
    # Заголовок
    st.title("🎓 Скользящая диагностика")
    
    # Настройка экзамена (боковая панель)
    setup_exam()
    
    # Прогресс и аналитика
    display_progress()
    display_analytics()
    
    # Основная область чата
    chat_container = st.container()
    
    with chat_container:
        # Отображение сообщений чата
        display_chat_messages()
        
        # Кнопка для генерации первого вопроса
        if st.session_state.exam_started and not st.session_state.get('first_question_generated', False) and not st.session_state.exam_completed:
            if st.button("🚀 Получить первый вопрос", type="primary"):
                with st.spinner("Генерация первого вопроса..."):
                    get_next_question()
                    st.session_state.first_question_generated = True
                st.rerun()
        
        # Поле ввода ответа
        elif st.session_state.waiting_for_answer and not st.session_state.exam_completed:
            with st.form("answer_form", clear_on_submit=True):
                user_answer = st.text_area(
                    "Ваш ответ:",
                    height=100,
                    placeholder="Введите ваш развернутый ответ здесь..."
                )
                
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    submit_button = st.form_submit_button("📤 Отправить ответ")
                with col2:
                    skip_button = st.form_submit_button("⏭️ Пропустить")
                
                if submit_button and user_answer.strip():
                    submit_answer(user_answer.strip())
                elif skip_button:
                    submit_answer("Ответ пропущен")
        
        # Кнопка сброса экзамена
        if st.session_state.exam_started:
            st.sidebar.markdown("---")
            if st.sidebar.button("🔄 Начать заново", type="secondary"):
                # Сброс состояния
                keys_to_keep = ['student_name']
                for key in list(st.session_state.keys()):
                    if key not in keys_to_keep:
                        del st.session_state[key]
                initialize_session_state()
                st.rerun()
    
    # Информация о системе
    if not st.session_state.exam_started:
        st.markdown("""
        ### 🤖 О системе
        
        Интеллектуальная система экзаменирования с адаптивными вопросами и детальным анализом ответов.
        
        **Возможности:**
        - 🎯 Адаптивные вопросы
        - 📊 Многокритериальная оценка
        - 💡 Персональные рекомендации
        - ✏️ Собственные темы экзаменов
        
        **Способы выбора темы:**
        - **Готовые темы** - выберите из предустановленных тем
        - **Своя тема** - создайте экзамен по любой теме
        
        Выберите тему в боковой панели и начните экзамен!
        """)

if __name__ == "__main__":
    main()
