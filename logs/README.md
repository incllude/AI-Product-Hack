# Логирование диалогов

Эта папка содержит логи диалогов экзаменационной системы.

## Структура логов

Каждый диалог сохраняется в отдельный JSON файл с именем:
```
dialog_YYYYMMDD_HHMMSS_sessionID.json
```

Где:
- `YYYYMMDD_HHMMSS` - дата и время начала диалога
- `sessionID` - уникальный 8-символьный идентификатор сессии

## Структура JSON файла

```json
{
  "session_info": {
    "session_id": "abc12345",
    "student_name": "Имя студента",
    "start_time": "2024-01-15T10:30:00.123456",
    "end_time": "2024-01-15T10:45:00.654321",
    "status": "completed",
    "duration_seconds": 900,
    "duration_formatted": "0:15:00.530865"
  },
  "exam_config": {
    "topic_info": {
      "name": "Название темы",
      "subject": "Предмет",
      "difficulty": "средний",
      "type": "predefined"
    },
    "max_questions": 5,
    "use_theme_structure": false
  },
  "messages": [
    {
      "timestamp": "2024-01-15T10:30:00.123456",
      "role": "assistant|user",
      "content": "Текст сообщения",
      "type": "text|question|evaluation",
      "metadata": {}
    }
  ],
  "questions_and_answers": [
    {
      "question": {
        "timestamp": "2024-01-15T10:30:00.123456",
        "question_number": 1,
        "question": "Текст вопроса",
        "topic_level": "базовый",
        "question_type": "open",
        "metadata": {}
      },
      "answer": {
        "timestamp": "2024-01-15T10:31:00.123456",
        "content": "Ответ студента"
      },
      "evaluation": {
        "timestamp": "2024-01-15T10:31:30.123456",
        "total_score": 8,
        "criteria_scores": {
          "correctness": 8,
          "completeness": 7,
          "understanding": 9
        },
        "strengths": "Хорошее понимание основ",
        "weaknesses": "Не хватает деталей",
        "metadata": {}
      }
    }
  ],
  "final_report": {
    "timestamp": "2024-01-15T10:45:00.123456",
    "report_data": {
      "grade_info": {
        "grade": "good",
        "percentage": 80,
        "points": "24/30",
        "description": "Хорошее понимание темы"
      },
      "recommendations": [
        "Рекомендация 1",
        "Рекомендация 2"
      ]
    }
  },
  "statistics": {
    "total_questions": 3,
    "total_answers": 3,
    "average_score": 7.8,
    "total_score": 24,
    "max_possible_score": 30
  }
}
```

## Использование логов

Логи можно использовать для:
- Анализа успеваемости студентов
- Улучшения алгоритмов оценки
- Исследования эффективности вопросов
- Отладки системы
- Восстановления сессий

## Конфиденциальность

⚠️ **Внимание**: Логи содержат персональные данные студентов. Обеспечьте соответствующую защиту файлов.
