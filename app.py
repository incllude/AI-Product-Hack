"""
Streamlit приложение-чатбот для интеллектуальной системы диагностики с LangGraph агентами
Адаптированная версия с поддержкой логирования и совместимостью с LangGraph
"""
import streamlit as st
import sys
import os
import json
import time
import uuid
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Добавляем путь к модулям агентов
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

from exam_orchestrator import ExamOrchestrator
from exam_orchestrator_optimized import OptimizedExamOrchestrator
from topic_manager import TopicManager

# Переключатель оптимизации - можно изменить на False для использования обычной версии
USE_OPTIMIZED_VERSION = True

class DialogLogger:
    """Класс для логирования диалогов экзамена - совместимый с LangGraph"""
    
    def __init__(self):
        self.logs_dir = os.path.join(os.path.dirname(__file__), 'logs', 'dialogs')
        os.makedirs(self.logs_dir, exist_ok=True)
        self.session_id = None
        self.log_file_path = None
        self.dialog_data = None
    
    def start_session(self, student_name, topic_info, max_questions, use_theme_structure):
        """Начало новой сессии диалога"""
        self.session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Имя файла: dialog_YYYYMMDD_HHMMSS_sessionID.json
        filename = f"dialog_{timestamp}_{self.session_id}.json"
        self.log_file_path = os.path.join(self.logs_dir, filename)
        
        # Инициализация структуры данных диалога
        self.dialog_data = {
            "session_info": {
                "session_id": self.session_id,
                "student_name": student_name,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "status": "started",
                "agent_type": "langgraph"  # Указываем тип агентов
            },
            "exam_config": {
                "topic_info": topic_info,
                "max_questions": max_questions,
                "use_theme_structure": use_theme_structure
            },
            "messages": [],
            "questions_and_answers": [],
            "evaluations": [],
            "final_report": None,
            "statistics": {
                "total_questions": 0,
                "total_answers": 0,
                "average_score": 0,
                "total_score": 0,
                "max_possible_score": 0
            },
            "langgraph_metadata": {  # Специфичные для LangGraph данные
                "workflow_state": None,
                "exam_session_id": None,
                "theme_structure": None
            }
        }
        
        self._save_log()
        return self.session_id
    
    def log_message(self, role, content, message_type="text", metadata=None):
        """Логирование сообщения в диалоге"""
        if not self.dialog_data:
            return
        
        message_entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "type": message_type,
            "metadata": metadata or {}
        }
        
        self.dialog_data["messages"].append(message_entry)
        self._save_log()
    
    def log_question(self, question_data):
        """Логирование вопроса - адаптировано для LangGraph структуры"""
        if not self.dialog_data:
            return
        
        # Адаптируем структуру вопроса из LangGraph формата
        question_entry = {
            "timestamp": datetime.now().isoformat(),
            "question_number": question_data.get('question_number', 0),
            "question": question_data.get('question', ''),
            "topic_level": question_data.get('topic_level', ''),
            "question_type": question_data.get('question_type', ''),
            "key_points": question_data.get('key_points', ''),
            "langgraph_metadata": {
                "generated_by": question_data.get('generated_by', 'unknown'),
                "theme_requirements": question_data.get('theme_requirements'),
                "raw_response": question_data.get('raw_response')
            },
            "metadata": question_data
        }
        
        self.dialog_data["questions_and_answers"].append({
            "question": question_entry,
            "answer": None,
            "evaluation": None
        })
        
        self.dialog_data["statistics"]["total_questions"] += 1
        self._save_log()
    
    def log_answer_and_evaluation(self, answer, evaluation_data):
        """Логирование ответа и его оценки - адаптировано для LangGraph"""
        if not self.dialog_data or not self.dialog_data["questions_and_answers"]:
            return
        
        # Находим последний вопрос без ответа
        for qa_pair in reversed(self.dialog_data["questions_and_answers"]):
            if qa_pair["answer"] is None:
                qa_pair["answer"] = {
                    "timestamp": datetime.now().isoformat(),
                    "content": answer
                }
                
                # Адаптируем структуру оценки для LangGraph
                qa_pair["evaluation"] = {
                    "timestamp": datetime.now().isoformat(),
                    "total_score": evaluation_data.get('total_score', 0),
                    "criteria_scores": evaluation_data.get('criteria_scores', {}),
                    "strengths": evaluation_data.get('strengths', ''),
                    "weaknesses": evaluation_data.get('weaknesses', ''),
                    "langgraph_metadata": {
                        "evaluation_type": evaluation_data.get('evaluation_type'),
                        "raw_evaluation": evaluation_data.get('raw_evaluation'),
                        "evaluation_summary": evaluation_data.get('evaluation_summary')
                    },
                    "metadata": evaluation_data
                }
                break
        
        # Обновляем статистику
        self.dialog_data["statistics"]["total_answers"] += 1
        
        # Пересчитываем статистику оценок
        evaluations = [qa["evaluation"] for qa in self.dialog_data["questions_and_answers"] if qa["evaluation"]]
        if evaluations:
            total_score = sum(eval_data["total_score"] for eval_data in evaluations)
            max_possible = len(evaluations) * 10  # Максимум 10 баллов за вопрос
            
            self.dialog_data["statistics"]["total_score"] = total_score
            self.dialog_data["statistics"]["max_possible_score"] = max_possible
            self.dialog_data["statistics"]["average_score"] = total_score / len(evaluations)
        
        self._save_log()
    
    def log_final_report(self, final_report):
        """Логирование финального отчета"""
        if not self.dialog_data:
            return
        
        self.dialog_data["final_report"] = {
            "timestamp": datetime.now().isoformat(),
            "report_data": final_report
        }
        
        self._save_log()
    
    def log_langgraph_session(self, exam_session_id, workflow_state=None):
        """Логирование специфичных для LangGraph данных"""
        if not self.dialog_data:
            return
        
        self.dialog_data["langgraph_metadata"]["exam_session_id"] = exam_session_id
        if workflow_state:
            # Сериализуем только сериализуемые части состояния
            serializable_state = {
                "session_id": workflow_state.get("session_id"),
                "status": workflow_state.get("status"),
                "current_question_number": workflow_state.get("current_question_number"),
                "max_questions": workflow_state.get("max_questions")
            }
            self.dialog_data["langgraph_metadata"]["workflow_state"] = serializable_state
        
        self._save_log()
    
    def end_session(self, status="completed"):
        """Завершение сессии диалога"""
        if not self.dialog_data:
            return
        
        self.dialog_data["session_info"]["end_time"] = datetime.now().isoformat()
        self.dialog_data["session_info"]["status"] = status
        
        # Вычисляем общую длительность сессии
        start_time = datetime.fromisoformat(self.dialog_data["session_info"]["start_time"])
        end_time = datetime.fromisoformat(self.dialog_data["session_info"]["end_time"])
        duration = (end_time - start_time).total_seconds()
        
        self.dialog_data["session_info"]["duration_seconds"] = duration
        self.dialog_data["session_info"]["duration_formatted"] = str(end_time - start_time)
        
        self._save_log()
    
    def _save_log(self):
        """Сохранение лога в файл"""
        if not self.dialog_data or not self.log_file_path:
            return
        
        try:
            # Создаем копию данных для сериализации
            data_to_save = self._prepare_data_for_json(self.dialog_data)
            
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка при сохранении лога: {e}")
    
    def _prepare_data_for_json(self, data):
        """Подготовка данных для JSON сериализации (конвертация datetime в строки)"""
        if isinstance(data, dict):
            return {key: self._prepare_data_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_data_for_json(item) for item in data]
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return data
    
    def get_session_summary(self):
        """Получение краткой сводки по текущей сессии"""
        if not self.dialog_data:
            return None
        
        return {
            "session_id": self.session_id,
            "student_name": self.dialog_data["session_info"]["student_name"],
            "topic_name": self.dialog_data["exam_config"]["topic_info"].get("name", "Unknown"),
            "questions_count": self.dialog_data["statistics"]["total_questions"],
            "answers_count": self.dialog_data["statistics"]["total_answers"],
            "average_score": self.dialog_data["statistics"]["average_score"],
            "log_file": self.log_file_path,
            "agent_type": "langgraph"
        }

# Конфигурация страницы
st.set_page_config(
    page_title="🎓 Скользящая диагностика (LangGraph)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    if 'dialog_logger' not in st.session_state:
        st.session_state.dialog_logger = None
    
    if 'final_report_generated' not in st.session_state:
        st.session_state.final_report_generated = False

def add_message(role, content, message_type="text", metadata=None):
    """Добавление сообщения в чат"""
    message = {
        "role": role,
        "content": content,
        "type": message_type,
        "timestamp": datetime.now()
    }
    
    st.session_state.messages.append(message)
    
    # Логирование сообщения
    if st.session_state.dialog_logger:
        st.session_state.dialog_logger.log_message(role, content, message_type, metadata)

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
    optimization_status = "⚡ Оптимизированная" if USE_OPTIMIZED_VERSION else "🔧 Стандартная"
    st.sidebar.caption(f"{optimization_status} версия: LangGraph агенты")
    
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
    use_theme_structure = st.sidebar.checkbox("Использовать структуру по Блуму", False)
    
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
    """Запуск экзамена с LangGraph агентами"""
    # Показываем индикатор загрузки
    with st.spinner("Инициализация экзамена (LangGraph)..."):
        try:
            # Инициализация логгера диалогов
            st.session_state.dialog_logger = DialogLogger()
            session_id = st.session_state.dialog_logger.start_session(
                st.session_state.student_name, 
                topic_info, 
                max_questions, 
                use_theme_structure
            )
            
            # Создание оркестратора LangGraph (оптимизированного или обычного)
            if USE_OPTIMIZED_VERSION:
                st.session_state.orchestrator = OptimizedExamOrchestrator(
                    topic_info=topic_info,
                    max_questions=max_questions,
                    use_theme_structure=use_theme_structure
                )
            else:
                st.session_state.orchestrator = ExamOrchestrator(
                    topic_info=topic_info,
                    max_questions=max_questions,
                    use_theme_structure=use_theme_structure
                )
            
            # Запуск экзамена
            session_info = st.session_state.orchestrator.start_exam(st.session_state.student_name)
            
            # Логирование LangGraph session
            if hasattr(st.session_state.orchestrator, 'current_session'):
                exam_session_id = getattr(st.session_state.orchestrator.current_session, 'session_id', None)
                st.session_state.dialog_logger.log_langgraph_session(exam_session_id)
            
            st.session_state.exam_started = True
            st.session_state.topic_selected = True
            st.session_state.exam_completed = False
            st.session_state.first_question_generated = False
            st.session_state.final_report_generated = False
            
            # Приветственное сообщение
            welcome_msg = f"""Добро пожаловать, {st.session_state.student_name}! 🎓

**Тема:** {topic_info['name']}
**Предмет:** {topic_info['subject']}
**Вопросов:** {max_questions}
**Режим:** {'Структурированный' if use_theme_structure else 'Быстрый'}
**ID сессии:** {session_id}
**Агенты:** LangGraph 🔧

Экзамен готов к началу! Нажмите кнопку ниже для получения первого вопроса."""
            
            add_message("assistant", welcome_msg, metadata={"session_id": session_id, "agent_type": "langgraph"})
            
        except Exception as e:
            st.error(f"Ошибка при запуске экзамена: {str(e)}")
            return
    
    st.rerun()

def get_next_question():
    """Получение следующего вопроса от LangGraph агентов"""
    if not st.session_state.orchestrator:
        return
    
    try:
        question_data = st.session_state.orchestrator.get_next_question()
        
        if 'question' in question_data:
            st.session_state.current_question = question_data
            st.session_state.waiting_for_answer = True
            
            # Логирование вопроса
            if st.session_state.dialog_logger:
                st.session_state.dialog_logger.log_question(question_data)
            
            # Форматирование вопроса для отображения
            question_text = f"""**Вопрос {question_data.get('question_number', '?')}**

{question_data['question']}

*Уровень: {question_data.get('topic_level', 'базовый')}*"""
            
            add_message("assistant", question_text, "question", metadata=question_data)
            
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
    """Отправка ответа на оценку LangGraph агентам"""
    if not st.session_state.orchestrator or not st.session_state.current_question:
        return
    
    # Добавляем ответ пользователя в чат
    add_message("user", answer)
    
    # Показываем индикатор загрузки для оценки
    with st.spinner("Оценка вашего ответа (LangGraph)..."):
        try:
            # Отправляем ответ на оценку
            evaluation = st.session_state.orchestrator.submit_answer(answer)
            
            # Логирование ответа и оценки
            if st.session_state.dialog_logger:
                st.session_state.dialog_logger.log_answer_and_evaluation(answer, evaluation)
            
            # Форматирование результата оценки
            eval_text = f"""**Оценка: {evaluation.get('total_score', 0)}/10 баллов**"""
            
            # Проверяем наличие предупреждения о согласованности
            evaluation_metadata = evaluation.get('evaluation_metadata', {})
            consistency_warning = evaluation_metadata.get('consistency_warning')
            
            if consistency_warning:
                eval_text += f"\n\n⚠️ **Внимание:** {consistency_warning}"
            
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
            
            # Показываем дополнительную информацию об оценке для отладки
            if evaluation_metadata:
                calculated_score = evaluation_metadata.get('calculated_score')
                llm_final_score = evaluation_metadata.get('llm_final_score')
                score_method = evaluation_metadata.get('score_method')
                
                if calculated_score is not None and llm_final_score is not None:
                    eval_text += f"\n🔍 **Детали оценки:** LLM: {llm_final_score}, Расчетная: {calculated_score}"
            
            if evaluation.get('strengths'):
                eval_text += f"\n\n**✅ Сильные стороны:** {evaluation['strengths']}"
            
            if evaluation.get('weaknesses'):
                eval_text += f"\n\n**❌ Области для улучшения:** {evaluation['weaknesses']}"
            
            add_message("assistant", eval_text, "evaluation", metadata=evaluation)
            
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
        # Экзамен завершен - генерируем финальный отчет напрямую
        st.session_state.exam_completed = True
        with st.spinner("Подготовка финального отчета..."):
            time.sleep(0.5)
            generate_final_report()
    
    st.rerun()

def generate_final_report():
    """Генерация финального отчета"""
    if not st.session_state.orchestrator:
        return
    
    # Защита от повторного вызова
    if st.session_state.get('final_report_generated', False):
        return
    
    try:
        # Получаем финальный отчет через complete_exam
        final_report = st.session_state.orchestrator.complete_exam()
        
        # Логирование финального отчета
        if st.session_state.dialog_logger:
            st.session_state.dialog_logger.log_final_report(final_report)
            st.session_state.dialog_logger.end_session("completed")
        
        if 'error' not in final_report:
            # Получаем краткую сводку сессии для отображения
            session_summary = None
            if st.session_state.dialog_logger:
                session_summary = st.session_state.dialog_logger.get_session_summary()
            
            report_text = f"""🎉 **Экзамен завершен!** (LangGraph агенты)

**Итоговые результаты:**
• Оценка: {final_report['grade_info']['grade'].upper()}
• Успеваемость: {final_report['grade_info']['percentage']}%
• Баллы: {final_report['grade_info']['points']}

**Описание:** {final_report['grade_info']['description']}

**Рекомендации:**"""
            
            for i, recommendation in enumerate(final_report['recommendations'][:3], 1):
                report_text += f"\n{i}. {recommendation}"
            
            if session_summary:
                report_text += f"\n\n📝 **Лог сессии сохранен:** `{os.path.basename(session_summary['log_file'])}`"
                report_text += f"\n🔧 **Тип агентов:** {session_summary['agent_type']}"
            
            add_message("assistant", report_text, metadata={"final_report": final_report, "agent_type": "langgraph"})
            # Устанавливаем флаг успешной генерации отчета
            st.session_state.final_report_generated = True
        else:
            add_message("assistant", f"❌ Ошибка при генерации отчета: {final_report.get('error', 'Неизвестная ошибка')}")
            # Даже при ошибке отмечаем попытку генерации, чтобы избежать повторов
            st.session_state.final_report_generated = True
    
    except Exception as e:
        st.error(f"Ошибка при генерации финального отчета: {str(e)}")
        # Отмечаем попытку генерации при исключении
        st.session_state.final_report_generated = True

def display_progress():
    """Отображение прогресса экзамена - адаптировано для LangGraph"""
    if not st.session_state.orchestrator or not st.session_state.exam_started:
        return
    
    try:
        progress = st.session_state.orchestrator.get_progress()
        
        st.sidebar.markdown("### 📊 Прогресс экзамена")
        st.sidebar.caption("🔧 LangGraph агенты")
        
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
    """Отображение аналитики в боковой панели - адаптировано для LangGraph"""
    if not st.session_state.orchestrator or not st.session_state.exam_started:
        return
    
    try:
        progress = st.session_state.orchestrator.get_progress()
        
        if progress['questions_answered'] > 0:
            st.sidebar.markdown("### 📈 Аналитика")
            
            # Получаем оценки из LangGraph session
            if hasattr(st.session_state.orchestrator, 'current_session') and st.session_state.orchestrator.current_session:
                evaluations = st.session_state.orchestrator.current_session.evaluations
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
    optimization_emoji = "⚡" if USE_OPTIMIZED_VERSION else "🔧"
    optimization_text = "Оптимизированная версия" if USE_OPTIMIZED_VERSION else "Стандартная версия"
    st.caption(f"{optimization_emoji} {optimization_text} - Powered by LangGraph агенты")
    
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
                with st.spinner("Генерация первого вопроса (LangGraph)..."):
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
            
            # Показываем информацию о текущей сессии
            if st.session_state.dialog_logger:
                session_summary = st.session_state.dialog_logger.get_session_summary()
                if session_summary:
                    st.sidebar.markdown("### 📝 Текущая сессия")
                    st.sidebar.write(f"**ID:** {session_summary['session_id']}")
                    st.sidebar.write(f"**Тип:** {session_summary['agent_type']}")
                    st.sidebar.write(f"**Вопросов:** {session_summary['questions_count']}")
                    st.sidebar.write(f"**Ответов:** {session_summary['answers_count']}")
                    if session_summary['answers_count'] > 0:
                        st.sidebar.write(f"**Средний балл:** {session_summary['average_score']:.1f}")
            
            if st.sidebar.button("🔄 Начать заново", type="secondary"):
                # Завершаем текущую сессию логирования
                if st.session_state.dialog_logger:
                    st.session_state.dialog_logger.end_session("reset")
                
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
        
        **Технологии:**
        - 🔧 **LangGraph агенты** - workflow-based архитектура
        - 📊 Многокритериальная оценка
        - 💡 Персональные рекомендации
        - 📝 Детальное логирование сессий
        
        **Возможности:**
        - 🎯 Адаптивные вопросы
        - ✏️ Собственные темы экзаменов
        - 🗂️ Тематическая структура по Блуму
        - 📈 Аналитика в реальном времени
        
        **Способы выбора темы:**
        - **Готовые темы** - выберите из предустановленных тем
        - **Своя тема** - создайте экзамен по любой теме
        
        Выберите тему в боковой панели и начните экзамен!
        """)

if __name__ == "__main__":
    main()
