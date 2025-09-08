#!/usr/bin/env python3
"""
Простой запускающий скрипт для демонстрации LangGraph системы
"""
import sys

def main():
    print("🚀 LANGGRAPH СИСТЕМА ЭКЗАМЕНАЦИОННЫХ АГЕНТОВ")
    print("=" * 50)
    
    # Запускаем основную демонстрацию
    try:
        print("\n🔄 Запуск example_usage.py...")
        
        # Импортируем и запускаем
        import example_usage
        example_usage.main_example()
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("\n💡 Возможные решения:")
        print("1. Установите зависимости: pip install -r requirements.txt")
        print("2. Убедитесь что langgraph установлен: pip install langgraph>=0.2.0")
        print("3. Проверьте наличие файла example_usage.py в текущей папке")
        
    except Exception as e:
        print(f"❌ Ошибка выполнения: {e}")
        print("\n🩺 Попробуйте запустить проверку системы:")
        print("   python example_usage.py")
        print("   Проверьте статус системы и зависимостей")

if __name__ == "__main__":
    main()
