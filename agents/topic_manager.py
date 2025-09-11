"""
Модуль для управления темами экзаменов
"""
from typing import Dict, List, Optional
import json


class TopicManager:
    """Менеджер тем для экзаменационной системы"""
    
    def __init__(self):
        """Инициализация менеджера тем"""
        self.predefined_topics = {
            "python_loops": {
                "name": "Циклы в Python",
                "subject": "Программирование",
                "description": "Изучение циклов for, while, break, continue в Python",
                "difficulty_levels": ["легкий", "средний", "сложный"],
                "sample_questions": [
                    "Объясните разницу между циклом for и while",
                    "Как работает оператор break в Python?",
                    "Что такое list comprehension и как его использовать?"
                ]
            },
            "photoeffect": {
                "name": "Законы фотоэффекта",
                "subject": "Физика",
                "description": "Изучение фотоэлектрического эффекта и его законов",
                "difficulty_levels": ["легкий", "средний", "сложный"],
                "sample_questions": [
                    "Сформулируйте законы фотоэффекта",
                    "Объясните уравнение Эйнштейна для фотоэффекта",
                    "Что такое красная граница фотоэффекта?"
                ]
            }
        }
    
    def get_predefined_topics(self) -> Dict[str, Dict]:
        """Возвращает список предопределенных тем"""
        return self.predefined_topics.copy()
    
    def get_topic_info(self, topic_key: str) -> Optional[Dict]:
        """
        Получает информацию о конкретной теме
        
        Args:
            topic_key: Ключ темы
            
        Returns:
            Информация о теме или None
        """
        return self.predefined_topics.get(topic_key)
    
    def display_topic_menu(self) -> None:
        """Отображает меню выбора тем"""
        print("Доступные темы экзамена:")
        print("=" * 50)
        
        for i, (key, topic) in enumerate(self.predefined_topics.items(), 1):
            print(f"{i}. {topic['name']}")
            print(f"   Предмет: {topic['subject']}")
            print(f"   Описание: {topic['description']}")
            print(f"   Уровни: {', '.join(topic['difficulty_levels'])}")
            print()
    
    def get_topic_selection(self) -> Dict[str, any]:
        """
        Интерактивный выбор темы экзамена
        
        Returns:
            Информация о выбранной теме
        """
        print("🎯 ВЫБОР ТЕМЫ ЭКЗАМЕНА")
        print("=" * 40)
        print("1. Выбрать из готовых тем")
        print("2. Ввести свою тему")
        
        choice = input("\nВаш выбор (1 или 2): ").strip()
        
        if choice == "1":
            return self._select_predefined_topic()
        elif choice == "2":
            return self._create_custom_topic()
        else:
            print("Неверный выбор. Используется тема по умолчанию.")
            return self._get_default_topic()
    
    def _select_predefined_topic(self) -> Dict[str, any]:
        """Выбор из предопределенных тем"""
        print("\n📚 ГОТОВЫЕ ТЕМЫ:")
        self.display_topic_menu()
        
        topic_keys = list(self.predefined_topics.keys())
        
        while True:
            try:
                choice = input(f"Выберите тему (1-{len(topic_keys)}): ").strip()
                topic_index = int(choice) - 1
                
                if 0 <= topic_index < len(topic_keys):
                    selected_key = topic_keys[topic_index]
                    topic_info = self.predefined_topics[selected_key]
                    
                    print(f"\n✅ Выбрана тема: {topic_info['name']}")
                    
                    # Выбор уровня сложности
                    difficulty = self._select_difficulty(topic_info['difficulty_levels'])
                    
                    return {
                        'type': 'predefined',
                        'key': selected_key,
                        'name': topic_info['name'],
                        'description': topic_info['description'],
                        'difficulty': difficulty
                    }
                else:
                    print("Неверный номер. Попробуйте еще раз.")
            except ValueError:
                print("Введите число. Попробуйте еще раз.")
    
    def _create_custom_topic(self) -> Dict[str, any]:
        """Создание пользовательской темы"""
        print("\n✏️ СОЗДАНИЕ СВОЕЙ ТЕМЫ:")
        
        name = input("Название темы: ").strip()
        if not name:
            name = "Пользовательская тема"
        
        subject = input("Предмет (по умолчанию 'Общие знания'): ").strip()
        if not subject:
            subject = "Общие знания"
        
        description = input("Краткое описание темы (необязательно): ").strip()
        if not description:
            description = f"Экзамен по теме: {name}"
        
        # Выбор уровня сложности
        print("\nДоступные уровни сложности:")
        print("1. Легкий")
        print("2. Средний")
        print("3. Сложный")
        
        difficulty = self._select_difficulty(["легкий", "средний", "сложный"])
        
        
        print(f"\n✅ Создана тема: {name}")
        
        return {
            'type': 'custom',
            'key': 'custom',
            'name': name,
            'subject': subject,
            'description': description,
            'difficulty': difficulty
        }
    
    def _select_difficulty(self, available_levels: List[str]) -> str:
        """Выбор уровня сложности"""
        print(f"\nВыберите уровень сложности:")
        for i, level in enumerate(available_levels, 1):
            print(f"{i}. {level.capitalize()}")
        
        while True:
            try:
                choice = input(f"Ваш выбор (1-{len(available_levels)}): ").strip()
                if not choice:
                    return "средний"  # По умолчанию
                
                level_index = int(choice) - 1
                if 0 <= level_index < len(available_levels):
                    return available_levels[level_index]
                else:
                    print("Неверный номер. Попробуйте еще раз.")
            except ValueError:
                print("Введите число. Попробуйте еще раз.")
    
    def _get_default_topic(self) -> Dict[str, any]:
        """Возвращает тему по умолчанию"""
        default_key = "python_loops"
        topic_info = self.predefined_topics[default_key]
        
        return {
            'type': 'predefined',
            'key': default_key,
            'name': topic_info['name'],
            'description': topic_info['description'],
            'difficulty': "средний"
        }
    
    def get_topic_context_for_prompts(self, topic_info: Dict[str, any]) -> str:
        """
        Создает контекст темы для промптов агентов
        
        Args:
            topic_info: Информация о теме
            
        Returns:
            Текстовый контекст для промптов
        """
        context = f"ТЕМА ЭКЗАМЕНА: {topic_info['name']}\n"
        context += f"ОПИСАНИЕ: {topic_info['description']}\n"
        context += f"УРОВЕНЬ СЛОЖНОСТИ: {topic_info['difficulty']}\n"
        
        
        context += "\nВопросы должны быть строго по данной теме и соответствовать указанному уровню сложности."
        
        return context
    
    def suggest_related_topics(self, custom_topic_name: str) -> List[str]:
        """
        Предлагает похожие темы на основе пользовательского ввода
        
        Args:
            custom_topic_name: Название пользовательской темы
            
        Returns:
            Список похожих тем
        """
        suggestions = []
        topic_name_lower = custom_topic_name.lower()
        
        # Простой поиск по ключевым словам
        if any(word in topic_name_lower for word in ['python', 'программирование', 'код', 'цикл']):
            suggestions.append("Циклы в Python")
        
        if any(word in topic_name_lower for word in ['физика', 'фотоэффект', 'эйнштейн', 'квант']):
            suggestions.append("Законы фотоэффекта")
        
        return suggestions
    
    def export_topic_info(self, topic_info: Dict[str, any], format: str = 'json') -> Dict[str, any]:
        """
        Экспортирует информацию о теме
        
        Args:
            topic_info: Информация о теме
            format: Формат экспорта
            
        Returns:
            Данные для экспорта
        """
        if format == 'json':
            return {
                'topic_export': topic_info,
                'export_timestamp': str(json.dumps(topic_info, ensure_ascii=False, indent=2))
            }
        else:
            return topic_info
    
    def validate_topic_info(self, topic_info: Dict[str, any]) -> bool:
        """
        Валидирует информацию о теме
        
        Args:
            topic_info: Информация о теме
            
        Returns:
            True если тема валидна
        """
        required_fields = ['name', 'difficulty']
        
        for field in required_fields:
            if field not in topic_info or not topic_info[field]:
                return False
        
        # Проверка уровня сложности
        valid_difficulties = ['легкий', 'средний', 'сложный']
        if topic_info['difficulty'] not in valid_difficulties:
            return False
        
        return True
