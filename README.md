# Реализация Target-Specific and Adversarial Prompt Injection against Code LLMs
Реализация на Python системы оптимизации триггеров для языковых моделей кода с использованием методов генерации на основе цепей Маркова и алгоритма Greedy Coordinate Gradient (GCG).

## Основные возможности

- Два метода генерации триггеров:
  - На основе цепей Маркова
  - Равномерная генерация
  - Greedy Coordinate Gradient (GCG)
- Оптимизация на основе градиентов
- Поддержка устройств CUDA и CPU
- Настраиваемые параметры оптимизации
- Расчет функции потерь с состовляющими противодействия

## Установка

```bash
pip install torch
pip install -U bitsandbytes
pip install accelerate
```

## Использование

### Базовый пример

```python
from trigger_optimizer import TriggerOptimizer

# Инициализация оптимизатора с вашей моделью и токенизатором
optimizer = TriggerOptimizer(model, tokenizer)

# Пример кода для оптимизации
source_code = "import os def calculate_sum(numbers):\n    "
target_code = "return sum(numbers)"

# Оптимизация триггера с использованием обоих методов
optimized_trigger = optimizer.optimize_trigger(
    source_code=source_code,
    target_code=target_code,
    method='both'  # Варианты: 'both', 'markov', 'uniform'
)
```

### Расширенный пример использования

```python
# Модифицируем входные данные
Ts = """
def calc_avg(numbers):\n return sum(numbers) / len(numbers)
"""

# Создание prompt с использованием оптимизированного триггера
prompt = f"""<PRE>
{' '.join(optimized_trigger)}

def calc_avg(numbers):
  return sum(numbers) / len(numbers)

{Ts}
</PRE>"""

# Генерация кода с использованием модели
with torch.no_grad():
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(optimizer.device)
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        max_length=200,
        num_return_sequences=3,  # Генерируем несколько вариантов
        temperature=0.9,         # Увеличили температуру для более разнообразных результатов
        top_p=0.92,
        do_sample=True,
        repetition_penalty=1.2,  # Добавили штраф за повторения
        pad_token_id=tokenizer.eos_token_id,
        num_beams=3             # Добавили beam search
    )

# Преобразование и вывод результатов
generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
for i, text in enumerate(generated_texts):
    print(f"\nGenerated text {i+1}:\n{text}")
```

## Параметры

- `method`: Метод генерации триггера ('markov', 'uniform' или 'both')
- `k`: Количество токенов для обновления за итерацию
- `max_iter`: Максимальное количество итераций оптимизации
- `device`: Вычислительное устройство ('cuda' или 'cpu')

## Компоненты

### MarkovChainGenerator

Генерирует токены, похожие на слова, используя подход на основе цепей Маркова.

### TriggerOptimizer

Основной класс, который обрабатывает:
- Начальную генерацию триггеров
- Процесс оптимизации
- Расчет функции потерь
- Вычисление градиентов
- Обновление токенов

## Методы

- `generate_initial_trigger()`: Создает начальные токены триггера
- `optimize_trigger()`: Основной цикл оптимизации
- `get_loss()`: Вычисляет функцию потерь для данного триггера
- `_update_trigger()`: Обновляет токены триггера на основе градиентов
- `_compute_gradient()`: Вычисляет градиенты для оптимизации

## Требования

- Python 3.6+
- PyTorch
- NumPy

## Лицензия

MIT

## Участие в разработке

Не стесняйтесь создавать issues и предлагать улучшения!

## Пример результатов

Оптимизатор способен генерировать эффективные триггеры для различных задач трансформации кода. Результаты могут варьироваться в зависимости от входных параметров и целевого кода.

## Примечание

Данный инструмент предназначен только для исследовательских целей. Пожалуйста, используйте его ответственно и в соответствии с этическими нормами.
