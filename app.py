"""
Streamlit приложение-чатбот для интеллектуальной системы диагностики
"""
import streamlit as st
import sys
import os
import json
import time
import uuid
import re
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
    """Класс для логирования диалогов экзамена"""
    
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
            "exam_metadata": {
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
        
        # Расширенные метаданные для сообщения
        enhanced_metadata = metadata or {}
        enhanced_metadata.update({
            "message_id": str(uuid.uuid4())[:8],
            "session_id": self.session_id,
            "message_length": len(content) if content else 0,
            "logged_at": datetime.now().isoformat()
        })
        
        message_entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "type": message_type,
            "metadata": enhanced_metadata
        }
        
        self.dialog_data["messages"].append(message_entry)
        self._save_log()
    
    def log_question(self, question_data):
        """Логирование вопроса"""
        if not self.dialog_data:
            return
        
        # Адаптируем структуру вопроса
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
    
    def log_answer_and_evaluation(self, answer, evaluation_data, response_time=None):
        """Логирование ответа и его оценки"""
        if not self.dialog_data or not self.dialog_data["questions_and_answers"]:
            return
        
        # Находим последний вопрос без ответа
        for qa_pair in reversed(self.dialog_data["questions_and_answers"]):
            if qa_pair["answer"] is None:
                qa_pair["answer"] = {
                    "timestamp": datetime.now().isoformat(),
                    "content": answer,
                    "response_time_seconds": response_time
                }
                
                # Адаптируем структуру оценки
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
        
        # Расширенная информация о финальном отчете
        enhanced_report = {
            "timestamp": datetime.now().isoformat(),
            "report_data": final_report,
            "summary": {
                "total_questions_logged": len(self.dialog_data.get("questions_and_answers", [])),
                "total_messages_logged": len(self.dialog_data.get("messages", [])),
                "session_duration_seconds": None,
                "completion_status": "completed_with_report"
            }
        }
        
        # Вычисляем длительность сессии, если возможно
        if self.dialog_data.get("session_info", {}).get("start_time"):
            try:
                start_time = datetime.fromisoformat(self.dialog_data["session_info"]["start_time"])
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                enhanced_report["summary"]["session_duration_seconds"] = duration
            except:
                pass
        
        self.dialog_data["final_report"] = enhanced_report
        self._save_log()
    
    def log_exam_session(self, exam_session_id, workflow_state=None):
        """Логирование данных экзаменационной сессии"""
        if not self.dialog_data:
            return
        
        self.dialog_data["exam_metadata"]["exam_session_id"] = exam_session_id
        if workflow_state:
            # Сериализуем только сериализуемые части состояния
            serializable_state = {
                "session_id": workflow_state.get("session_id"),
                "status": workflow_state.get("status"),
                "current_question_number": workflow_state.get("current_question_number"),
                "max_questions": workflow_state.get("max_questions")
            }
            self.dialog_data["exam_metadata"]["workflow_state"] = serializable_state
        
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
    page_title="🎓 Скользящая диагностика",
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
    
    # Таймер для ответов
    if 'question_timer_start' not in st.session_state:
        st.session_state.question_timer_start = None
    
    if 'answer_time_limit' not in st.session_state:
        st.session_state.answer_time_limit = 60  # 60 секунд по умолчанию (1 минута)
    
    if 'timer_expired' not in st.session_state:
        st.session_state.timer_expired = False
    
    if 'last_timer_update' not in st.session_state:
        st.session_state.last_timer_update = 0
    
    if 'auto_skip_triggered' not in st.session_state:
        st.session_state.auto_skip_triggered = False
    
    # Состояния для просмотра деталей диагностики
    if 'viewing_exam_details' not in st.session_state:
        st.session_state.viewing_exam_details = False
    
    if 'selected_exam_filename' not in st.session_state:
        st.session_state.selected_exam_filename = None
    
    if 'viewing_final_report' not in st.session_state:
        st.session_state.viewing_final_report = False

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

def check_timer_expiry():
    """Проверка истечения времени для ответа"""
    if (st.session_state.waiting_for_answer and 
        st.session_state.question_timer_start is not None and 
        not st.session_state.timer_expired):
        
        elapsed_time = time.time() - st.session_state.question_timer_start
        if elapsed_time >= st.session_state.answer_time_limit:
            st.session_state.timer_expired = True
            return True
    return False

def get_remaining_time():
    """Получение оставшегося времени в секундах"""
    if (st.session_state.waiting_for_answer and 
        st.session_state.question_timer_start is not None and 
        not st.session_state.timer_expired):
        
        elapsed_time = time.time() - st.session_state.question_timer_start
        remaining_time = max(0, st.session_state.answer_time_limit - elapsed_time)
        return remaining_time
    return 0

def display_timer():
    """Отображение таймера обратного отсчета"""
    if (st.session_state.waiting_for_answer and 
        st.session_state.question_timer_start is not None and 
        not st.session_state.timer_expired):
        
        remaining_time = get_remaining_time()
        
        if remaining_time > 0:
            minutes = int(remaining_time // 60)
            seconds = int(remaining_time % 60)
            
            # Определяем цвет в зависимости от оставшегося времени
            if remaining_time > 30:
                color = "#4CAF50"  # Зеленый
            elif remaining_time > 10:
                color = "#FF9800"  # Оранжевый
            else:
                color = "#f44336"  # Красный
            
            # HTML для красивого таймера
            timer_html = f"""
            <div style="
                background: linear-gradient(135deg, {color} 0%, {color}dd 100%);
                color: white;
                padding: 15px 20px;
                border-radius: 15px;
                text-align: center;
                margin: 10px 0;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                font-size: 18px;
                font-weight: bold;
            ">
                ⏱️ Осталось времени: {minutes:02d}:{seconds:02d}
            </div>
            """
            st.markdown(timer_html, unsafe_allow_html=True)
            
            # Если время истекло, показываем предупреждение
            if remaining_time <= 5:
                st.warning("⚠️ Время почти истекло! Поторопитесь с ответом!")
            
        else:
            st.error("⏰ Время истекло! Вопрос будет автоматически пропущен.")

def display_chat_messages():
    """Отображение истории чата"""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.write("**👤 Ваш ответ:**")
            st.write(message['content'])
            st.divider()
        
        elif message["role"] == "assistant":
            if message.get("type") == "question":
                # Извлекаем номер вопроса из контента
                content = message['content']
                # Ищем "Вопрос X" в начале контента
                question_match = re.match(r'\*\*Вопрос (\d+|\?)\*\*\n\n(.+)', content, re.DOTALL)
                if question_match:
                    question_number = question_match.group(1)
                    question_text = question_match.group(2)
                    st.write(f"**❓ Вопрос {question_number}:**")
                    st.write(question_text)
                else:
                    # Если не удалось распарсить, показываем как есть
                    st.write("**❓ Вопрос:**")
                    st.write(content)
                st.divider()
                
            
            elif message.get("type") == "evaluation":
                st.write("**📊 Оценка:**")
                st.write(message['content'])
                st.divider()
            
            else:
                st.info(message['content'])
                st.divider()

def load_exam_history():
    """Загрузка истории диагностик из логов"""
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs', 'dialogs')
    history = []
    
    if not os.path.exists(logs_dir):
        return history
    
    try:
        for filename in os.listdir(logs_dir):
            if filename.endswith('.json') and filename.startswith('dialog_'):
                filepath = os.path.join(logs_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    session_info = data.get('session_info', {})
                    exam_config = data.get('exam_config', {})
                    statistics = data.get('statistics', {})
                    
                    history_item = {
                        'filename': filename,
                        'session_id': session_info.get('session_id', 'Unknown'),
                        'student_name': session_info.get('student_name', 'Unknown'),
                        'topic_name': exam_config.get('topic_info', {}).get('name', 'Unknown'),
                        'start_time': session_info.get('start_time', ''),
                        'questions_count': statistics.get('total_questions', 0),
                        'average_score': statistics.get('average_score', 0),
                        'agent_type': session_info.get('agent_type', 'unknown')
                    }
                    history.append(history_item)
                except Exception as e:
                    continue
    except Exception as e:
        st.error(f"Ошибка при загрузке истории: {str(e)}")
    
    # Сортируем по времени (новые сверху)
    history.sort(key=lambda x: x['start_time'], reverse=True)
    return history

def load_exam_details(filename):
    """Загрузка детальных данных диагностики"""
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs', 'dialogs')
    filepath = os.path.join(logs_dir, filename)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Ошибка при загрузке данных диагностики: {str(e)}")
        return None

def display_embedded_final_report(exam_data):
    """Отображение встроенного финального отчета в детальном просмотре"""
    # Используем существующую функцию визуализации, но с компактным заголовком
    data = extract_historical_evaluation_data(exam_data)
    if not data or data['questions_count'] == 0:
        st.info("📊 Данные для анализа недоступны")
        return
    
    # Компактный заголовок для встроенного отчета
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        text-align: center;
    ">
        <h3 style="margin: 0; color: white;">📊 Результаты диагностики</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Основная статистика
    avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
    max_score = max(data['scores']) if data['scores'] else 0
    min_score = min(data['scores']) if data['scores'] else 0
    
    # Метрики в полном формате
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    with col_metric1:
        st.metric("🎯 Средний балл", f"{avg_score:.1f}/10")
    with col_metric2:
        st.metric("⭐ Лучший результат", f"{max_score}/10")
    with col_metric3:
        st.metric("📈 Вопросов пройдено", f"{data['questions_count']}")
    with col_metric4:
        success_rate = (avg_score / 10) * 100
        st.metric("📊 Успеваемость", f"{success_rate:.0f}%")
    
    # Дополнительная строка метрик
    col_metric5, col_metric6, col_metric7, col_metric8 = st.columns(4)
    with col_metric5:
        st.metric("📉 Худший результат", f"{min_score}/10")
    with col_metric6:
        # Пустая колонка для симметрии
        pass
    with col_metric7:
        # Пустая колонка для симметрии
        pass
    with col_metric8:
        # Пустая колонка для симметрии
        pass
    
    st.markdown("---")
    
    # Создаем колонки для графиков (компактная версия)
    col1, col2 = st.columns(2)
    
    with col1:
        # График оценок по критериям
        if any(data['criteria'].values()):
            st.markdown("**📈 Оценки по критериям**")
            
            criteria_names = {
                'correctness': 'Правильность',
                'completeness': 'Полнота',
                'understanding': 'Понимание'
            }
            
            # Создаем DataFrame для радиальной диаграммы
            criteria_avg = {}
            for criterion, scores in data['criteria'].items():
                if scores:
                    criteria_avg[criteria_names.get(criterion, criterion)] = sum(scores) / len(scores)
            
            if criteria_avg:
                # Радиальная диаграмма (компактная)
                categories = list(criteria_avg.keys())
                values = list(criteria_avg.values())
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Результаты',
                    line=dict(color='#4CAF50', width=3),
                    fillcolor='rgba(76, 175, 80, 0.3)'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10],
                            tickmode='linear',
                            tick0=0,
                            dtick=2
                        )
                    ),
                    height=300,
                    showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # График динамики баллов
        if data['scores']:
            st.markdown("**📉 Динамика баллов**")
            question_numbers = list(range(1, len(data['scores']) + 1))
                    
            fig = px.line(
                x=question_numbers, 
                y=data['scores'],
                title="Баллы по вопросам",
                labels={'x': 'Номер вопроса', 'y': 'Баллы'},
                markers=True
            )
            fig.update_traces(
                line=dict(color='#4CAF50', width=3),
                marker=dict(size=8, color='#45a049')
            )
            fig.update_layout(
                height=300, 
                yaxis=dict(range=[0, 10]),
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Создаем колонки для анализа сильных и слабых сторон
    col_analysis1, col_analysis2 = st.columns(2)
    
    with col_analysis1:
        # Анализ сильных сторон
        if data['strengths']:
            st.subheader("✅ Сильные стороны")
            
            # Создаем интерактивную визуализацию
            strength_counts = {}
            for item in data['strengths']:
                # Простая группировка по ключевым словам
                strength = item['strength'].lower()
                if 'понимание' in strength or 'понимает' in strength:
                    strength_counts['Понимание концепций'] = strength_counts.get('Понимание концепций', 0) + 1
                elif 'правильн' in strength or 'корректн' in strength or 'точн' in strength:
                    strength_counts['Правильность ответов'] = strength_counts.get('Правильность ответов', 0) + 1
                elif 'объяснение' in strength or 'изложение' in strength:
                    strength_counts['Качество объяснения'] = strength_counts.get('Качество объяснения', 0) + 1
                else:
                    strength_counts['Общие знания'] = strength_counts.get('Общие знания', 0) + 1
            
            if strength_counts:
                # Горизонтальная гистограмма
                categories = list(strength_counts.keys())
                values = list(strength_counts.values())
                
                fig = px.bar(
                    x=values,
                    y=categories,
                    orientation='h',
                    title="Распределение сильных сторон",
                    color=values,
                    color_continuous_scale='Greens'
                )
                fig.update_layout(height=250, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Список сильных сторон (компактный)
            for item in data['strengths'][:3]:  # Показываем только первые 3
                with st.expander(f"🌟 {item['question']} (балл: {item['score']})"):
                    st.write(item['strength'])
            
            if len(data['strengths']) > 3:
                st.caption(f"И еще {len(data['strengths']) - 3} сильных сторон...")
    
    with col_analysis2:
        # Анализ слабых сторон
        if data['weaknesses']:
            st.subheader("❌ Области для улучшения")
            
            # Группировка слабых сторон
            weakness_counts = {}
            for item in data['weaknesses']:
                weakness = item['weakness'].lower()
                if 'пример' in weakness or 'иллюстрац' in weakness:
                    weakness_counts['Недостаток примеров'] = weakness_counts.get('Недостаток примеров', 0) + 1
                elif 'детал' in weakness or 'подробн' in weakness:
                    weakness_counts['Неполнота ответа'] = weakness_counts.get('Неполнота ответа', 0) + 1
                elif 'объяснение' in weakness or 'изложение' in weakness:
                    weakness_counts['Качество изложения'] = weakness_counts.get('Качество изложения', 0) + 1
                else:
                    weakness_counts['Прочие недочеты'] = weakness_counts.get('Прочие недочеты', 0) + 1
            
            if weakness_counts:
                # Горизонтальная гистограмма
                categories = list(weakness_counts.keys())
                values = list(weakness_counts.values())
                
                fig = px.bar(
                    x=values,
                    y=categories,
                    orientation='h',
                    title="Распределение слабых сторон",
                    color=values,
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=250, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Список слабых сторон (компактный)
            for item in data['weaknesses'][:3]:  # Показываем только первые 3
                with st.expander(f"⚠️ {item['question']} (балл: {item['score']})"):
                    st.write(item['weakness'])
            
            if len(data['weaknesses']) > 3:
                st.caption(f"И еще {len(data['weaknesses']) - 3} областей для улучшения...")
    
    # Дополнительная секция с рекомендациями
    st.markdown("---")
    st.subheader("💡 Рекомендации для дальнейшего развития")
    
    if avg_score >= 8.5:
        st.success("🏆 **Отличный результат!** Вы демонстрируете глубокое понимание темы. Рекомендуем перейти к более сложным темам или применить знания на практике.")
    elif avg_score >= 7.0:
        st.info("👍 **Хороший результат!** Основы освоены хорошо. Стоит поработать над деталями и добавить больше практических примеров.")
    elif avg_score >= 5.0:
        st.warning("📚 **Удовлетворительный результат.** Есть понимание основ, но нужна дополнительная практика и изучение материала.")
    else:
        st.error("💪 **Нужно больше практики!** Рекомендуем повторить основы темы и пройти диагностику еще раз.")

def display_exam_details(exam_data, item):
    """Отображение детальной информации о диагностике"""
    if not exam_data:
        st.error("Не удалось загрузить данные диагностики")
        return
    
    # Убираем отдельный режим просмотра финального отчета - теперь он всегда показывается сверху
    
    # Заголовок страницы детального просмотра
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
    ">
        <h1 style="margin: 0; color: white;">📋 Детали диагностики</h1>
        <h3 style="margin: 0.5rem 0; color: white;">{exam_data.get('exam_config', {}).get('topic_info', {}).get('name', 'Неизвестная тема')}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Информация о сессии
    session_info = exam_data.get('session_info', {})
    exam_config = exam_data.get('exam_config', {})
    statistics = exam_data.get('statistics', {})
    
    # Колонки для общей информации
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("👤 Студент", session_info.get('student_name', 'Неизвестно'))
        try:
            from datetime import datetime
            start_time = datetime.fromisoformat(session_info.get('start_time', '').replace('Z', '+00:00'))
            formatted_time = start_time.strftime("%d.%m.%Y %H:%M")
            st.metric("📅 Дата и время", formatted_time)
        except:
            st.metric("📅 Дата и время", session_info.get('start_time', '')[:16])
    
    with col2:
        topic_info = exam_config.get('topic_info', {})
        difficulty = topic_info.get('difficulty', 'Неизвестно')
        st.metric("🎯 Сложность", difficulty.capitalize())
        exam_mode = "Структурированный" if exam_config.get('use_theme_structure', False) else "Быстрый"
        st.metric("⚙️ Режим", exam_mode)
    
    with col3:
        st.metric("❓ Вопросов", f"{statistics.get('total_questions', 0)}")
        if statistics.get('average_score', 0) > 0:
            st.metric("⭐ Средний балл", f"{statistics.get('average_score', 0):.1f}/10")
        else:
            st.metric("⭐ Средний балл", "Нет данных")
    
    st.markdown("---")
    
    # Показываем финальный отчет с визуализацией сразу после основной информации
    final_report = exam_data.get('final_report', {})
    questions_and_answers = exam_data.get('questions_and_answers', [])
    has_evaluations = any(qa.get('evaluation') for qa in questions_and_answers)
    
    # Если есть данные для отчета, показываем его
    if (final_report and final_report.get('report_data')) or has_evaluations:
        # Создаем компактную версию финального отчета для встраивания
        display_embedded_final_report(exam_data)
        st.markdown("---")
    
    # Вопросы и ответы
    
    if not questions_and_answers:
        st.warning("Вопросы и ответы не найдены в данной диагностике")
        return
    
    st.subheader("❓ Вопросы и ответы")
    
    for i, qa_pair in enumerate(questions_and_answers, 1):
        question_data = qa_pair.get('question', {})
        answer_data = qa_pair.get('answer', {})
        evaluation_data = qa_pair.get('evaluation', {})
        
        # Используем expander для каждого вопроса
        with st.expander(f"Вопрос {i}: {question_data.get('question', 'Вопрос не найден')[:80]}...", expanded=(i == 1)):
            
            # Вопрос
            st.markdown("### 📝 Вопрос")
            st.write(question_data.get('question', 'Текст вопроса не найден'))
            
            # Дополнительная информация о вопросе
            if question_data.get('topic_level'):
                st.caption(f"**Уровень:** {question_data.get('topic_level')}")
            
            # Ответ
            st.markdown("### 💬 Ответ студента")
            if answer_data:
                answer_content = answer_data.get('content', 'Ответ не найден')
                if answer_content == "Ответ пропущен" or answer_content == "Ответ пропущен (время истекло)":
                    st.warning(f"⏭️ {answer_content}")
                else:
                    st.write(answer_content)
                
                # Время ответа
                if answer_data.get('response_time_seconds'):
                    response_time = answer_data.get('response_time_seconds')
                    minutes = int(response_time // 60)
                    seconds = int(response_time % 60)
                    st.caption(f"⏱️ Время ответа: {minutes}:{seconds:02d}")
            else:
                st.warning("Ответ не найден")
            
            # Оценка
            st.markdown("### 📊 Оценка")
            if evaluation_data:
                # Общий балл
                total_score = evaluation_data.get('total_score', 0)
                st.metric("🎯 Общий балл", f"{total_score}/10")
                
                # Оценки по критериям
                criteria_scores = evaluation_data.get('criteria_scores', {})
                if criteria_scores:
                    st.markdown("**📈 Оценки по критериям:**")
                    criteria_cols = st.columns(len(criteria_scores))
                    criteria_names = {
                        'correctness': 'Правильность',
                        'completeness': 'Полнота',
                        'understanding': 'Понимание'
                    }
                    
                    for idx, (criterion, score) in enumerate(criteria_scores.items()):
                        with criteria_cols[idx]:
                            criterion_name = criteria_names.get(criterion, criterion)
                            st.metric(criterion_name, f"{score}/10")
                
                # Сильные стороны
                if evaluation_data.get('strengths'):
                    st.markdown("**✅ Сильные стороны:**")
                    st.success(evaluation_data.get('strengths'))
                
                # Слабые стороны
                if evaluation_data.get('weaknesses'):
                    st.markdown("**❌ Области для улучшения:**")
                    st.warning(evaluation_data.get('weaknesses'))
                
            else:
                st.warning("Оценка не найдена")
            
            st.markdown("---")
    
    # Кнопка возврата
    st.markdown("---")
    col_back1, col_back2, col_back3 = st.columns([1, 2, 1])
    with col_back2:
        if st.button("🔙 Назад к списку диагностик", type="primary", use_container_width=True):
            st.session_state['viewing_exam_details'] = False
            st.session_state['selected_exam_filename'] = None
            st.session_state['viewing_final_report'] = False
            st.rerun()

def display_exam_history():
    """Отображение истории диагностик в sidebar"""
    st.sidebar.header("📚 История диагностик")
    
    history = load_exam_history()
    
    if not history:
        st.sidebar.info("История пуста")
        return
    
    # Ограничиваем количество показываемых записей
    max_display = 10
    display_history = history[:max_display]
    
    for item in display_history:
        # Определяем индикатор на основе данных
        if item['questions_count'] > 0 and item['average_score'] > 0:
            status_emoji = '🟢'  # Завершено с результатами
        elif item['questions_count'] > 0:
            status_emoji = '🟡'  # Есть вопросы, но нет оценок
        else:
            status_emoji = '⚪'  # Начато, но мало данных
        
        # Форматируем время
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(item['start_time'].replace('Z', '+00:00'))
            time_str = dt.strftime("%d.%m %H:%M")
        except:
            time_str = item['start_time'][:16]
        
        with st.sidebar.expander(f"{status_emoji} {item['topic_name'][:20]}... ({time_str})"):
            st.write(f"**Студент:** {item['student_name']}")
            st.write(f"**Вопросов:** {item['questions_count']}")
            if item['average_score'] > 0:
                st.write(f"**Средний балл:** {item['average_score']:.1f}")
            
            # Кнопка для просмотра детального лога
            if st.button("📄 Подробнее", key=f"detail_{item['session_id']}", help="Просмотр детального лога"):
                # Устанавливаем состояние для просмотра деталей
                st.session_state['viewing_exam_details'] = True
                st.session_state['selected_exam_filename'] = item['filename']
                st.session_state['viewing_final_report'] = False  # Сбрасываем при переходе
                st.rerun()
    
    if len(history) > max_display:
        st.sidebar.caption(f"Показано {max_display} из {len(history)} записей")
    
    # Дополнительные опции управления историей
    if history:
        st.sidebar.markdown("---")
        if st.sidebar.button("🗂️ Управление историей", help="Просмотр и управление всей историей диагностик"):
            st.sidebar.info("Функция управления историей будет добавлена позже")

def setup_exam_on_main():
    """Настройка экзамена на главном экране"""
    # Добавляем стильный заголовок
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    ">
        <h2 style="color: white; margin: 0;">⚙️ Настройка новой диагностики</h2>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Имя студента
    student_name = st.text_input("Имя студента", value=st.session_state.student_name, key="main_student_name")
    if student_name != st.session_state.student_name:
        st.session_state.student_name = student_name
    
    # Выбор способа задания темы
    topic_source = st.radio(
        "Способ выбора темы:",
        options=["Готовые темы", "Своя тема"],
        index=0,
        horizontal=True
    )
    
    # Используем колонки для лучшего макета
    col1, col2 = st.columns([2, 1])
    
    topic_manager = TopicManager()
    
    # Инициализируем переменные по умолчанию
    custom_name = ""
    custom_concepts_input = ""  # Initialize here to avoid UnboundLocalError
    
    with col1:
        if topic_source == "Готовые темы":
        # Выбор из предопределенных тем
            topics = topic_manager.get_predefined_topics()
        
            topic_options = {f"{topic['name']}": key 
                            for key, topic in topics.items()}
            
            selected_topic_display = st.selectbox(
                "Выберите тему экзамена",
                options=list(topic_options.keys()),
                index=0
            )
        
            selected_topic_key = topic_options[selected_topic_display]
            raw_topic = topics[selected_topic_key]
            
            # Создаем правильную структуру topic_info с полем difficulty
            topic_info = {
                'type': 'predefined',
                'key': selected_topic_key,
                'name': raw_topic['name'],
                'description': raw_topic['description'],
                'difficulty': 'средний',  # Установим по умолчанию
                'key_concepts': []
            }
    
        else:
        # Ввод пользовательской темы
            custom_name = st.text_input(
                "Название темы:",
                placeholder="Например: Квантовая физика"
            )
        
        
            custom_description = st.text_area(
                "Описание темы (необязательно):",
                placeholder="Краткое описание того, что будет изучаться...",
                height=80
            )
        
            
            # Создаем структуру для пользовательской темы
            topic_info = {
                'type': 'custom',
                'key': 'custom',
                'name': custom_name if custom_name.strip() else "Пользовательская тема",
                'description': custom_description if custom_description.strip() else f"Экзамен по теме: {custom_name if custom_name.strip() else 'Пользовательская тема'}",
                'difficulty': 'средний',
                'key_concepts': []
            }
    
    with col2:
    # Настройки экзамена
        st.subheader("Параметры")
        
        # Выбор уровня сложности
        difficulty_options = ["легкий", "средний", "сложный"]
        selected_difficulty = st.selectbox(
            "Уровень сложности",
            options=difficulty_options,
            index=1  # средний по умолчанию
        )
        topic_info['difficulty'] = selected_difficulty
        
        max_questions = st.slider("Количество вопросов", 3, 10, 5)
        use_theme_structure = st.checkbox("Использовать структуру по Блуму", False)
        
        # Настройка времени на ответ
        answer_time_minutes = st.slider("Время на ответ (минуты)", 1, 5, 1, step=1)
        answer_time_limit = answer_time_minutes * 60  # Конвертируем в секунды
        st.session_state.answer_time_limit = answer_time_limit
        
    
    # Проверка готовности к началу экзамена
    can_start_exam = True
    if topic_source == "Своя тема":
        if not custom_name.strip():
            can_start_exam = False
            st.warning("⚠️ Введите название темы")
    
    # Кнопка начала экзамена
    st.markdown("---")
    exam_disabled = st.session_state.exam_started or not can_start_exam
    
    col_start1, col_start2, col_start3 = st.columns([1, 2, 1])
    with col_start2:
        # Стилизованная кнопка запуска
        button_html = """
        <style>
        .start-button {
            background: linear-gradient(90deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 15px 30px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            width: 100%;
            text-align: center;
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
            transition: all 0.3s ease;
        }
        .start-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
        }
        .start-button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        </style>
        """
        st.markdown(button_html, unsafe_allow_html=True)
        
        if st.button("🚀 Начать диагностику", disabled=exam_disabled, type="primary", use_container_width=True):
            start_exam(topic_info, max_questions, use_theme_structure)
    
    return topic_info

def start_exam(topic_info, max_questions, use_theme_structure):
    """Запуск экзамена"""
    # Показываем индикатор загрузки
    with st.spinner("Инициализация экзамена..."):
        try:
            # Инициализация логгера диалогов
            st.session_state.dialog_logger = DialogLogger()
            session_id = st.session_state.dialog_logger.start_session(
                st.session_state.student_name, 
                topic_info, 
                max_questions, 
                use_theme_structure
            )
            
            # Создание оркестратора экзамена
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
            
            # Логирование экзаменационной сессии
            if hasattr(st.session_state.orchestrator, 'current_session'):
                exam_session_id = getattr(st.session_state.orchestrator.current_session, 'session_id', None)
                st.session_state.dialog_logger.log_exam_session(exam_session_id)
            
            st.session_state.exam_started = True
            st.session_state.topic_selected = True
            st.session_state.exam_completed = False
            st.session_state.first_question_generated = False
            st.session_state.final_report_generated = False
            
            # Приветственное сообщение
            welcome_msg = f"""Добро пожаловать, {st.session_state.student_name}! 🎓

**Тема:** {topic_info['name']}
**Вопросов:** {max_questions}
**Режим:** {'Структурированный' if use_theme_structure else 'Быстрый'}

Экзамен готов к началу! Нажмите кнопку ниже для получения первого вопроса."""
            
            add_message("assistant", welcome_msg, metadata={"session_id": session_id, "agent_type": "langgraph"})
            
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
            
            # Запуск таймера для ответа
            st.session_state.question_timer_start = time.time()
            st.session_state.timer_expired = False
            st.session_state.last_timer_update = 0  # Сбрасываем время последнего обновления
            st.session_state.auto_skip_triggered = False  # Сбрасываем флаг автопропуска
            
            # Логирование вопроса
            if st.session_state.dialog_logger:
                st.session_state.dialog_logger.log_question(question_data)
            
            # Форматирование вопроса для отображения
            question_text = f"""**Вопрос {question_data.get('question_number', '?')}**

{question_data['question']}"""
            
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
    """Отправка ответа на оценку"""
    if not st.session_state.orchestrator or not st.session_state.current_question:
        return
    
    # Сбрасываем состояние ожидания ответа и таймер
    st.session_state.waiting_for_answer = False
    st.session_state.question_timer_start = None
    st.session_state.timer_expired = False
    st.session_state.last_timer_update = 0
    st.session_state.auto_skip_triggered = False
    
    # Добавляем ответ пользователя в чат
    add_message("user", answer)
    
    # Показываем индикатор загрузки для оценки
    with st.spinner("Оценка вашего ответа..."):
        try:
            # Отправляем ответ на оценку
            evaluation = st.session_state.orchestrator.submit_answer(answer)
            
            # Вычисляем время ответа
            response_time = None
            if st.session_state.question_timer_start:
                response_time = time.time() - st.session_state.question_timer_start
            
            # Логирование ответа и оценки
            if st.session_state.dialog_logger:
                st.session_state.dialog_logger.log_answer_and_evaluation(answer, evaluation, response_time)
            
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
            
            report_text = f"""🎉 **Экзамен завершен!**

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

def display_progress_header():
    """Отображение компактного прогресса в шапке главного экрана"""
    if not st.session_state.orchestrator or not st.session_state.exam_started:
        return
    
    try:
        progress = st.session_state.orchestrator.get_progress()
        progress_percentage = (progress['questions_answered'] / progress['max_questions']) * 100
        
        # Создаем красивый прогресс-бар в шапке
        if progress['max_possible_score'] > 0:
            success_rate = (progress['current_score'] / progress['max_possible_score']) * 100
        else:
            success_rate = 0
            
        # Определяем цвет прогресса в зависимости от успеваемости
        if success_rate >= 80:
            progress_color = "#4CAF50"  # Зеленый
        elif success_rate >= 60:
            progress_color = "#FF9800"  # Оранжевый
        else:
            progress_color = "#f44336"  # Красный
            
        # Создаем фиксированный прогресс-бар с использованием специального контейнера
        progress_container = st.container()
        
        # Добавляем CSS и HTML для фиксированного прогресс-бара
        st.markdown(f"""
        <style>
        /* Убираем отступы у основного контейнера Streamlit */
        .main .block-container {{
            padding-top: 1rem;
        }}
        
        /* Отступ для sidebar чтобы не залезал под прогресс-бар */
        .css-1d391kg {{
            padding-top: 120px !important;
        }}
        
        /* Альтернативный селектор для sidebar */
        .css-1cypcdb {{
            padding-top: 120px !important;
        }}
        
        /* Еще один селектор для sidebar */
        .css-1v0mbdj {{
            padding-top: 120px !important;
        }}
        
        /* Универсальный селектор для sidebar */
        [data-testid="stSidebar"] {{
            padding-top: 120px !important;
        }}
        
        /* Селектор для контента sidebar */
        [data-testid="stSidebar"] > div {{
            padding-top: 120px !important;
        }}
        
        /* Фиксированный прогресс-бар */
        .fixed-progress-bar {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem 1.5rem;
            color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            z-index: 999999;
            border-bottom: 2px solid rgba(255,255,255,0.2);
        }}
        
        /* Отступ для контента под фиксированным баром - уменьшен, так как заголовок уже имеет отступ */
        .content-with-fixed-progress {{
            margin-top: 20px;
        }}
        
        .progress-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.8rem;
            flex-wrap: wrap;
            gap: 10px;
        }}
        
        .progress-left {{
            flex: 1;
        }}
        
        
        .progress-title {{
            margin: 0;
            color: white;
            font-size: 1.3rem;
            font-weight: bold;
        }}
        
        .progress-stats {{
            display: flex;
            gap: 16px;
            font-size: 16px;
            font-weight: 600;
            flex-wrap: wrap;
        }}
        
        .progress-stat {{
            background: rgba(255,255,255,0.25);
            padding: 8px 12px;
            border-radius: 10px;
            white-space: nowrap;
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .progress-bar-container {{
            background: rgba(255,255,255,0.2);
            border-radius: 25px;
            height: 10px;
            overflow: hidden;
            position: relative;
        }}
        
        .progress-bar {{
            background: linear-gradient(90deg, {progress_color} 0%, {progress_color}dd 100%);
            height: 100%;
            width: {progress_percentage}%;
            border-radius: 25px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }}
        
        .progress-bar::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%);
            animation: progressShine 2s infinite;
        }}
        
        @keyframes progressShine {{
            0% {{ transform: translateX(-100%); }}
            100% {{ transform: translateX(100%); }}
        }}
        
        .success-indicator {{
            color: #4CAF50;
            font-weight: bold;
        }}
        
        .warning-indicator {{
            color: #FF9800;
            font-weight: bold;
        }}
        
        .danger-indicator {{
            color: #f44336;
            font-weight: bold;
        }}
        
        /* Адаптивность для мобильных устройств */
        @media (max-width: 768px) {{
            .fixed-progress-bar {{
                padding: 0.8rem 1rem;
            }}
            .progress-stats {{
                font-size: 14px;
                gap: 12px;
            }}
            .progress-title {{
                font-size: 1.1rem;
            }}
            .content-with-fixed-progress {{
                margin-top: 20px;
            }}
            .main-title-with-progress {{
                margin-top: 160px !important;
            }}
            .progress-header {{
                flex-direction: column;
                gap: 8px;
                align-items: stretch;
            }}
            .progress-left {{
                flex: none;
            }}
        }}
        </style>
        
        <div class="fixed-progress-bar">
            <div class="progress-header">
                <div class="progress-left">
                    <h3 class="progress-title">📊 Прогресс диагностики</h3>
                    <div class="progress-stats">
                        <span class="progress-stat">📝 Вопрос {progress['questions_answered']}/{progress['max_questions']}</span>
                        <span class="progress-stat">⭐ Баллы {progress['current_score']}/{progress['max_possible_score']}</span>
                        <span class="progress-stat {'success-indicator' if success_rate >= 80 else 'warning-indicator' if success_rate >= 60 else 'danger-indicator'}">
                            📈 {success_rate:.0f}% успеваемость
                        </span>
                    </div>
                </div>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        
    except Exception as e:
        st.error(f"Ошибка при отображении прогресса: {str(e)}")


def extract_evaluation_data():
    """Извлечение данных оценок для визуализации из текущей сессии"""
    if not st.session_state.orchestrator or not hasattr(st.session_state.orchestrator, 'current_session'):
        return None
    
    session = st.session_state.orchestrator.current_session
    if not session or not session.evaluations:
        return None
    
    # Собираем данные из оценок
    strengths_data = []
    weaknesses_data = []
    criteria_data = {'correctness': [], 'completeness': [], 'understanding': []}
    scores = []
    
    for i, eval_data in enumerate(session.evaluations, 1):
        # Баллы
        scores.append(eval_data.get('total_score', 0))
        
        # Оценки по критериям
        criteria = eval_data.get('criteria_scores', {})
        for criterion, score in criteria.items():
            if criterion in criteria_data:
                criteria_data[criterion].append(score)
        
        # Сильные стороны
        strengths = eval_data.get('strengths', '')
        if strengths:
            strengths_data.append({
                'question': f"Вопрос {i}",
                'strength': strengths,
                'score': eval_data.get('total_score', 0)
            })
        
        # Слабые стороны
        weaknesses = eval_data.get('weaknesses', '')
        if weaknesses:
            weaknesses_data.append({
                'question': f"Вопрос {i}",
                'weakness': weaknesses,
                'score': eval_data.get('total_score', 0)
            })
    
    return {
        'strengths': strengths_data,
        'weaknesses': weaknesses_data,
        'criteria': criteria_data,
        'scores': scores,
        'questions_count': len(session.evaluations)
    }

def extract_historical_evaluation_data(exam_data):
    """Извлечение данных оценок для визуализации из сохраненных логов"""
    if not exam_data:
        return None
    
    questions_and_answers = exam_data.get('questions_and_answers', [])
    if not questions_and_answers:
        return None
    
    # Собираем данные из оценок
    strengths_data = []
    weaknesses_data = []
    criteria_data = {'correctness': [], 'completeness': [], 'understanding': []}
    scores = []
    
    for i, qa_pair in enumerate(questions_and_answers, 1):
        evaluation_data = qa_pair.get('evaluation', {})
        if not evaluation_data:
            continue
            
        # Баллы
        total_score = evaluation_data.get('total_score', 0)
        scores.append(total_score)
        
        # Оценки по критериям
        criteria_scores = evaluation_data.get('criteria_scores', {})
        for criterion, score in criteria_scores.items():
            if criterion in criteria_data:
                criteria_data[criterion].append(score)
        
        # Сильные стороны
        strengths = evaluation_data.get('strengths', '')
        if strengths:
            strengths_data.append({
                'question': f"Вопрос {i}",
                'strength': strengths,
                'score': total_score
            })
        
        # Слабые стороны
        weaknesses = evaluation_data.get('weaknesses', '')
        if weaknesses:
            weaknesses_data.append({
                'question': f"Вопрос {i}",
                'weakness': weaknesses,
                'score': total_score
            })
    
    return {
        'strengths': strengths_data,
        'weaknesses': weaknesses_data,
        'criteria': criteria_data,
        'scores': scores,
        'questions_count': len([qa for qa in questions_and_answers if qa.get('evaluation')])
    }

def display_results_visualization(exam_data=None):
    """Отображение визуализации результатов завершенной диагностики"""
    # Если переданы исторические данные, используем их, иначе берем из текущей сессии
    if exam_data:
        data = extract_historical_evaluation_data(exam_data)
        title_text = "📊 Анализ результатов диагностики"
    else:
        data = extract_evaluation_data()
        title_text = "📊 Анализ ваших результатов"
    
    if not data or data['questions_count'] == 0:
        st.warning("📊 Данные для визуализации недоступны")
        return
    
    # Красивый заголовок с результатами  
    header_title = "🎉 Диагностика завершена!" if not exam_data else "📊 Результаты диагностики"
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0;
        color: white;
        text-align: center;
    ">
        <h1 style="margin: 0; color: white;">{header_title}</h1>
        <h3 style="margin: 0.5rem 0; color: white;">{title_text}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Общая статистика
    avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
    max_score = max(data['scores']) if data['scores'] else 0
    min_score = min(data['scores']) if data['scores'] else 0
    
    # Метрики в шапке
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    with col_metric1:
        st.metric("🎯 Средний балл", f"{avg_score:.1f}/10")
    with col_metric2:
        st.metric("⭐ Лучший результат", f"{max_score}/10")
    with col_metric3:
        st.metric("📈 Вопросов пройдено", f"{data['questions_count']}")
    with col_metric4:
        success_rate = (avg_score / 10) * 100
        st.metric("📊 Успеваемость", f"{success_rate:.0f}%")
    
    st.markdown("---")
    
    # Создаем колонки для графиков
    col1, col2 = st.columns(2)
    
    with col1:
        # График оценок по критериям
        if any(data['criteria'].values()):
            st.subheader("📈 Оценки по критериям")
            
            criteria_names = {
                'correctness': 'Правильность',
                'completeness': 'Полнота',
                'understanding': 'Понимание'
            }
            
            # Создаем DataFrame для радиальной диаграммы
            criteria_avg = {}
            for criterion, scores in data['criteria'].items():
                if scores:
                    criteria_avg[criteria_names.get(criterion, criterion)] = sum(scores) / len(scores)
            
            if criteria_avg:
                # Радиальная диаграмма
                categories = list(criteria_avg.keys())
                values = list(criteria_avg.values())
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Ваши результаты',
                    line=dict(color='#667eea', width=3),
                    fillcolor='rgba(102, 126, 234, 0.3)'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10],
                            tickmode='linear',
                            tick0=0,
                            dtick=2
                        )
                    ),
                    height=400,
                    title="Профиль компетенций",
                    title_x=0.5
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # График динамики баллов
        if data['scores']:
            st.subheader("📉 Динамика баллов")
            question_numbers = list(range(1, len(data['scores']) + 1))
                    
            fig = px.line(
                x=question_numbers, 
                y=data['scores'],
                title="Баллы по вопросам",
                labels={'x': 'Номер вопроса', 'y': 'Баллы'},
                markers=True
            )
            fig.update_traces(
                line=dict(color='#4CAF50', width=3),
                marker=dict(size=8, color='#45a049')
            )
            fig.update_layout(height=300, yaxis=dict(range=[0, 10]))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Анализ сильных сторон
        if data['strengths']:
            st.subheader("✅ Сильные стороны")
            
            # Создаем интерактивную визуализацию
            strength_counts = {}
            for item in data['strengths']:
                # Простая группировка по ключевым словам
                strength = item['strength'].lower()
                if 'понимание' in strength or 'понимает' in strength:
                    strength_counts['Понимание концепций'] = strength_counts.get('Понимание концепций', 0) + 1
                elif 'правильн' in strength or 'корректн' in strength or 'точн' in strength:
                    strength_counts['Правильность ответов'] = strength_counts.get('Правильность ответов', 0) + 1
                elif 'объяснение' in strength or 'изложение' in strength:
                    strength_counts['Качество объяснения'] = strength_counts.get('Качество объяснения', 0) + 1
                else:
                    strength_counts['Общие знания'] = strength_counts.get('Общие знания', 0) + 1
            
            if strength_counts:
                # Горизонтальная гистограмма
                categories = list(strength_counts.keys())
                values = list(strength_counts.values())
                
                fig = px.bar(
                    x=values,
                    y=categories,
                    orientation='h',
                    title="Распределение сильных сторон",
                    color=values,
                    color_continuous_scale='Greens'
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Список сильных сторон
            for item in data['strengths']:
                with st.expander(f"🌟 {item['question']} (балл: {item['score']})"):
                    st.write(item['strength'])
        
        # Анализ слабых сторон
        if data['weaknesses']:
            st.subheader("❌ Области для улучшения")
            
            # Группировка слабых сторон
            weakness_counts = {}
            for item in data['weaknesses']:
                weakness = item['weakness'].lower()
                if 'пример' in weakness or 'иллюстрац' in weakness:
                    weakness_counts['Недостаток примеров'] = weakness_counts.get('Недостаток примеров', 0) + 1
                elif 'детал' in weakness or 'подробн' in weakness:
                    weakness_counts['Неполнота ответа'] = weakness_counts.get('Неполнота ответа', 0) + 1
                elif 'объяснение' in weakness or 'изложение' in weakness:
                    weakness_counts['Качество изложения'] = weakness_counts.get('Качество изложения', 0) + 1
                else:
                    weakness_counts['Прочие недочеты'] = weakness_counts.get('Прочие недочеты', 0) + 1
            
            if weakness_counts:
                # Горизонтальная гистограмма
                categories = list(weakness_counts.keys())
                values = list(weakness_counts.values())
                
                fig = px.bar(
                    x=values,
                    y=categories,
                    orientation='h',
                    title="Распределение слабых сторон",
                    color=values,
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Список слабых сторон
            for item in data['weaknesses']:
                with st.expander(f"⚠️ {item['question']} (балл: {item['score']})"):
                    st.write(item['weakness'])
    
    # Дополнительная секция с рекомендациями
    st.markdown("---")
    st.subheader("💡 Рекомендации для дальнейшего развития")
    
    if avg_score >= 8.5:
        st.success("🏆 **Отличный результат!** Вы демонстрируете глубокое понимание темы. Рекомендуем перейти к более сложным темам или применить знания на практике.")
    elif avg_score >= 7.0:
        st.info("👍 **Хороший результат!** Основы освоены хорошо. Стоит поработать над деталями и добавить больше практических примеров.")
    elif avg_score >= 5.0:
        st.warning("📚 **Удовлетворительный результат.** Есть понимание основ, но нужна дополнительная практика и изучение материала.")
    else:
        st.error("💪 **Нужно больше практики!** Рекомендуем повторить основы темы и пройти диагностику еще раз.")
    
    # Кнопки действий - адаптируем для исторических данных
    if exam_data:
        # Для исторических данных - только кнопка возврата
        col_back_vis = st.columns([1, 2, 1])[1]
        with col_back_vis:
            if st.button("🔙 Назад к деталям диагностики", type="primary", use_container_width=True):
                # Просто обновляем страницу - состояние просмотра деталей уже установлено
                st.rerun()
    else:
        # Для текущей сессии - обычные кнопки
        col_action1, col_action2, col_action3 = st.columns(3)
        with col_action1:
            if st.button("🔄 Пройти диагностику заново", type="secondary", use_container_width=True):
                # Завершаем текущую сессию логирования
                if st.session_state.dialog_logger:
                    st.session_state.dialog_logger.end_session("restart")
                
                # Сброс состояния
                keys_to_keep = ['student_name']
                for key in list(st.session_state.keys()):
                    if key not in keys_to_keep:
                        del st.session_state[key]
                initialize_session_state()
                st.rerun()
        
        with col_action2:
            if st.button("📄 Посмотреть историю", type="secondary", use_container_width=True):
                st.info("💡 История диагностик доступна в левой боковой панели")
        
        with col_action3:
            if st.button("📊 Скрыть анализ", type="secondary", use_container_width=True):
                st.session_state['hide_visualization'] = True
                st.rerun()


def main():
    """Основная функция приложения"""
    initialize_session_state()
    
    # Если просматриваем детали диагностики, показываем их
    if st.session_state.get('viewing_exam_details', False) and st.session_state.get('selected_exam_filename'):
        # История диагностик в sidebar (показываем всегда)
        display_exam_history()
        
        exam_data = load_exam_details(st.session_state.selected_exam_filename)
        if exam_data:
            # Найдем информацию о диагностике для передачи в функцию отображения
            history = load_exam_history()
            item = None
            for hist_item in history:
                if hist_item['filename'] == st.session_state.selected_exam_filename:
                    item = hist_item
                    break
            
            display_exam_details(exam_data, item)
            return  # Завершаем выполнение функции, чтобы не показывать основной интерфейс
    
    # Проверяем таймер в самом начале для автообновления
    if (st.session_state.waiting_for_answer and 
        not st.session_state.timer_expired and 
        st.session_state.question_timer_start is not None):
        
        # Если время истекло, имитируем нажатие кнопки "Пропустить"
        if check_timer_expiry():
            add_message("assistant", "⏰ Время на ответ истекло! Вопрос автоматически пропущен.")
            st.session_state.auto_skip_triggered = True
            st.rerun()
    
    # Заголовок с отступом если экзамен активен
    if st.session_state.exam_started:
        # Добавляем отступ сверху для заголовка, чтобы он не перекрывался с фиксированным прогресс-баром
        st.markdown("""
        <style>
        .main-title-with-progress {
            margin-top: 140px !important;
            margin-bottom: 0.5rem;
        }
        </style>
        <div class="main-title-with-progress">
        """, unsafe_allow_html=True)
        
    # Создаем колонки для заголовка и кнопки перезапуска (только во время экзамена)
    if st.session_state.exam_started:
        col_title, col_restart = st.columns([3, 1])
        with col_title:
            st.title("🎓 Скользящая диагностика")
        with col_restart:
            st.markdown("<br>", unsafe_allow_html=True)  # Добавляем отступ для выравнивания
            if st.button("🔄 Начать заново", type="secondary", key="main_restart", help="Завершить текущую диагностику и начать новую"):
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
    else:
        # Обычный заголовок когда экзамен не начат
        st.title("🎓 Скользящая диагностика")
    
    if st.session_state.exam_started:
        st.markdown('</div>', unsafe_allow_html=True)
    
    # История диагностик в sidebar
    display_exam_history()
    
    # Прогресс в шапке главного экрана (всегда виден)
    if st.session_state.exam_started:
        display_progress_header()
    
    
    # Основная область
    if not st.session_state.exam_started:
        # Показываем настройки новой диагностики на главном экране
        setup_exam_on_main()
        
        # Информация о системе
        st.markdown("""
        ### 🤖 О системе
        
        Интеллектуальная система экзаменирования с адаптивными вопросами и детальным анализом ответов.
        
        **Технологии:**
        - 🔧 **Интеллектуальные агенты** - современная архитектура
        - 📊 Многокритериальная оценка
        - 💡 Персональные рекомендации
        - 📝 Детальное логирование сессий
        
        **Возможности:**
        - 🎯 Адаптивные вопросы
        - ⏱️ Ограничение времени на ответ
        - ✏️ Собственные темы экзаменов
        - 🗂️ Тематическая структура по Блуму
        - 📈 Аналитика в реальном времени
        - 📚 Просмотр истории всех диагностик
        - 📄 Детальный анализ каждой диагностики
        
        **Способы выбора темы:**
        - **Готовые темы** - выберите из предустановленных тем
        - **Своя тема** - создайте экзамен по любой теме
        
        **Просмотр истории диагностик:**
        - 📚 В левой панели отображается история всех пройденных диагностик
        - 📄 Нажмите "Подробнее" для просмотра результатов, вопросов и ответов
        - 📊 Финальный отчет с графиками отображается автоматически сверху
        - 🔙 Кнопка "Назад к списку диагностик" для возврата к истории
        
        Настройте параметры диагностики выше и нажмите "Начать диагностику"!
        """)
    
    else:
        # Экзамен начат - показываем чат с отступом для фиксированного прогресс-бара
        st.markdown('<div class="content-with-fixed-progress">', unsafe_allow_html=True)
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
            # Проверяем истечение времени (дублирующая проверка для надежности)
            if check_timer_expiry():
                # Время истекло - устанавливаем флаг автопропуска
                add_message("assistant", "⏰ Время на ответ истекло! Вопрос автоматически пропущен.")
                st.session_state.auto_skip_triggered = True
                st.rerun()
            
            # Отображаем таймер
            display_timer()
            
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
                
                # Проверяем автоматический пропуск или нажатие кнопок
                if st.session_state.get('auto_skip_triggered', False):
                    # Автоматический пропуск - имитируем нажатие кнопки "Пропустить"
                    st.session_state.auto_skip_triggered = False
                    submit_answer("Ответ пропущен (время истекло)")
                elif submit_button and user_answer.strip():
                    submit_answer(user_answer.strip())
                elif skip_button:
                    submit_answer("Ответ пропущен")
        
        # Визуализация результатов (показываем только когда экзамен завершен)
        elif st.session_state.exam_completed and st.session_state.get('final_report_generated', False):
            # Проверяем, не скрыта ли визуализация
            if not st.session_state.get('hide_visualization', False):
                display_results_visualization()
            else:
                # Показываем кнопку для повторного показа анализа
                st.markdown("---")
                col_show1, col_show2, col_show3 = st.columns([1, 2, 1])
                with col_show2:
                    if st.button("📊 Показать анализ результатов", type="primary", use_container_width=True):
                        st.session_state['hide_visualization'] = False
                st.rerun()
    
        st.markdown('</div>', unsafe_allow_html=True)  # Закрываем div для отступа
    
    # Автообновление для таймера - принудительное обновление
    if (st.session_state.waiting_for_answer and 
        not st.session_state.timer_expired and 
        st.session_state.question_timer_start is not None):
        
        remaining_time = get_remaining_time()
        if remaining_time > 0:
            # Принудительное обновление каждые 2 секунды
            time.sleep(2)
            st.rerun()

if __name__ == "__main__":
    main()
