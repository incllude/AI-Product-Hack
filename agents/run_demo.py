#!/usr/bin/env python3
"""
Простой запускающий скрипт для демонстрации LangGraph системы
"""
import sys
import os

def main():
    print("🚀 LANGGRAPH СИСТЕМА ЭКЗАМЕНАЦИОННЫХ АГЕНТОВ")
    print("=" * 50)
    
    # Проверяем что мы в правильной папке
    current_dir = os.path.basename(os.getcwd())
    if current_dir != "langgraph_version":
        print("⚠️  ВНИМАНИЕ: Запустите скрипт из папки langgraph_version/")
        print(f"   Текущая папка: {current_dir}")
        print(f"   Ожидается: langgraph_version")
        
        choice = input("\nПопробовать перейти в правильную папку? (да/нет): ").strip().lower()
        if choice in ['да', 'yes', 'y', 'д']:
            try:
                os.chdir('langgraph_version')
                print("✅ Перешли в папку langgraph_version/")
            except:
                print("❌ Не удалось перейти в папку langgraph_version/")
                print("   Запустите скрипт вручную из правильной папки")
                return
        else:
            return
    
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
        print("3. Проверьте структуру файлов в папке langgraph_version/")
        
    except Exception as e:
        print(f"❌ Ошибка выполнения: {e}")
        print("\n🩺 Попробуйте запустить проверку системы:")
        print("   python example_usage_langgraph.py")
        print("   Выберите опцию 6 (Проверка статуса системы)")

if __name__ == "__main__":
    main()
