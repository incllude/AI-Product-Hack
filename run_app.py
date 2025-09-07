"""
Скрипт запуска Streamlit приложения
"""
import subprocess
import sys
import os

def main():
    """Запуск Streamlit приложения"""
    print("🎓 Запуск приложения 'Скользящая диагностика'...")
    print("=" * 50)
    
    # Проверка наличия .env файла
    if not os.path.exists('.env'):
        print("⚠️  ВНИМАНИЕ: Файл .env не найден!")
        print("Создайте файл .env с следующим содержимым:")
        print("YAGPT_API_KEY=ваш_api_ключ")
        print("YC_FOLDER_ID=ваш_folder_id")
        print("=" * 50)
    
    # Запуск Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Приложение остановлено пользователем")
    except Exception as e:
        print(f"❌ Ошибка при запуске: {e}")

if __name__ == "__main__":
    main()
