import torch                    # работа с тензорами
import torch.nn as nn            # слои нейронных сетей с обучаемыми параметрами
import torch.nn.functional as F  # функции без параметров (softmax и т.д.)
import math                      # нужен для math.sqrt


def scaled_dot_product_attention(Q, K, V, mask=None):
    # Та же функция что в attention.py — см. там подробные комментарии
    # Формула: softmax(QK^T / √d_k) × V

    d_k = Q.size(-1)
    # Берём последнее измерение Q — это размерность d_k

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # Матрица схожести Q с K, нормализованная на √d_k

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
        # Запрещённые позиции → -inf → после softmax станут 0

    weights = F.softmax(scores, dim=-1)
    # Превращаем оценки в веса (сумма по каждой строке = 1)

    return torch.matmul(weights, V), weights
    # Возвращаем взвешенную сумму V и сами веса (для визуализации)


class MultiHeadAttention(nn.Module):
    # Класс наследует nn.Module — это базовый класс для всех слоёв в PyTorch
    # Наследование даёт: автоматическое отслеживание параметров, .to(device), .train()/.eval()

    def __init__(self, d_model, h):
        # __init__ — конструктор, вызывается при создании объекта
        # d_model — полная размерность модели (например 512)
        # h       — количество голов (например 8)

        super().__init__()
        # Вызываем конструктор родителя (nn.Module) — обязательно для корректной работы

        assert d_model % h == 0, "d_model должен делиться на h"
        # assert — проверка условия; если False — выбрасывает ошибку
        # d_model=512, h=8 → 512/8=64 — ок
        # d_model=512, h=7 → 512/7≈73.1 — не целое → ошибка
        # Нужно чтобы каждая голова получила одинаковый кусок

        self.h = h
        # Сохраняем количество голов — понадобится в split_heads

        self.d_k = d_model // h
        # Размерность одной головы: 512 // 8 = 64
        # // — целочисленное деление (без остатка)
        # Каждая голова работает с d_k=64 вместо d_model=512

        self.W_Q = nn.Linear(d_model, d_model)
        # nn.Linear(in, out) — полносвязный слой: out = x @ W^T + b
        # Обучаемая матрица весов W_Q формы (d_model, d_model)
        # Проецирует входной вектор в пространство запросов
        # Почему d_model→d_model, а не d_model→d_k?
        #   Вместо h отдельных матриц (d_model→d_k) используем одну большую (d_model→d_model)
        #   Потом просто разрезаем результат на h кусков по d_k — это быстрее на GPU

        self.W_K = nn.Linear(d_model, d_model)
        # Аналогичная матрица для ключей

        self.W_V = nn.Linear(d_model, d_model)
        # Аналогичная матрица для значений

        self.W_O = nn.Linear(d_model, d_model)
        # Финальная матрица: после concat всех голов → проецируем обратно
        # Позволяет головам "смешивать" информацию между собой

    def split_heads(self, x):
        # Разрезаем d_model на h голов
        # Вход x: (batch, seq_len, d_model)
        # Выход:  (batch, h, seq_len, d_k)

        batch, seq_len, d_model = x.size()
        # Распаковываем три измерения тензора в три переменные

        x = x.view(batch, seq_len, self.h, self.d_k)
        # .view() — меняет форму тензора БЕЗ копирования данных
        # (batch, seq_len, d_model) → (batch, seq_len, h, d_k)
        # d_model = h * d_k, поэтому это просто "разрезаем" последнее измерение
        # Например: (2, 10, 512) → (2, 10, 8, 64)

        return x.transpose(1, 2)
        # Меняем местами измерения 1 и 2
        # (batch, seq_len, h, d_k) → (batch, h, seq_len, d_k)
        # Зачем: attention работает с (batch, heads, seq_len, d_k)
        # Так все 8 голов обрабатываются параллельно одним matmul

    def combine_heads(self, x):
        # Обратная операция: склеиваем головы обратно
        # Вход:  (batch, h, seq_len, d_k)
        # Выход: (batch, seq_len, d_model)

        batch, h, seq_len, d_k = x.size()
        # Распаковываем четыре измерения

        x = x.transpose(1, 2)
        # (batch, h, seq_len, d_k) → (batch, seq_len, h, d_k)
        # Возвращаем seq_len на позицию 1

        return x.contiguous().view(batch, seq_len, h * d_k)
        # .contiguous() — после transpose данные в памяти могут быть несмежными
        # .view() требует смежных данных → .contiguous() создаёт смежную копию
        # .view(batch, seq_len, h * d_k) — склеиваем h*d_k обратно в d_model
        # (batch, seq_len, h, d_k) → (batch, seq_len, d_model)

    def forward(self, Q, K, V, mask=None):
        # forward — вызывается при mha(Q, K, V)
        # PyTorch автоматически вызывает forward когда объект вызывается как функция
        # Q, K, V: (batch, seq_len, d_model) — входные данные

        Q = self.split_heads(self.W_Q(Q))
        # self.W_Q(Q) — применяем линейную проекцию: (batch, seq_len, d_model) → (batch, seq_len, d_model)
        # self.split_heads(...) — разрезаем на головы: → (batch, h, seq_len, d_k)
        # Теперь Q содержит проецированные запросы для всех голов

        K = self.split_heads(self.W_K(K))
        # Аналогично для ключей

        V = self.split_heads(self.W_V(V))
        # Аналогично для значений

        x, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # Запускаем attention для всех голов ПАРАЛЛЕЛЬНО
        # PyTorch обрабатывает измерение h как батч → все головы считаются одновременно
        # x: (batch, h, seq_len, d_k)
        # self.attn_weights — сохраняем веса чтобы можно было посмотреть снаружи

        x = self.combine_heads(x)
        # Склеиваем головы: (batch, h, seq_len, d_k) → (batch, seq_len, d_model)

        return self.W_O(x)
        # Финальная проекция: (batch, seq_len, d_model) → (batch, seq_len, d_model)
        # W_O позволяет модели "перемешать" информацию от разных голов
        # Форма входа = форма выхода — слой ничего не меняет структурно


if __name__ == "__main__":
    torch.manual_seed(42)
    # Фиксируем seed для воспроизводимости результатов

    batch, seq_len, d_model, h = 2, 10, 512, 8
    # batch=2    — 2 предложения в батче
    # seq_len=10 — 10 токенов в каждом
    # d_model=512 — размерность как в оригинальной статье
    # h=8        — 8 голов как в оригинальной статье

    mha = MultiHeadAttention(d_model=d_model, h=h)
    # Создаём объект — вызывается __init__, инициализируются веса W_Q, W_K, W_V, W_O

    x = torch.randn(batch, seq_len, d_model)
    # Случайный входной тензор (batch=2, seq_len=10, d_model=512)
    # Имитирует эмбеддинги токенов + позиционное кодирование

    out = mha(x, x, x)
    # Self-attention: Q=K=V=x (один и тот же вход)
    # В encoder всегда self-attention — каждый токен смотрит на все токены включая себя
    # mha(x, x, x) → вызывает mha.forward(x, x, x)

    print(f"d_model={d_model}, h={h}, d_k per head={d_model//h}")
    # Показываем конфигурацию: d_k = 512/8 = 64

    print(f"Вход:  {x.shape}")
    # (2, 10, 512)

    print(f"Выход: {out.shape}")
    # (2, 10, 512) — форма не изменилась!
    # Multi-head attention не меняет форму тензора, только "обновляет" содержимое

    print(f"Веса attention: {mha.attn_weights.shape}")
    # (2, 8, 10, 10) — для каждого из 8 голов матрица 10×10
    # mha.attn_weights[b, h, i, j] = "вес токена j для токена i в голове h в примере b"

    print("\n--- Веса разных голов для токена 0 ---")
    for head_i in range(4):
        w = mha.attn_weights[0, head_i, 0].detach()
        # [0] — первый пример в батче
        # [head_i] — текущая голова
        # [0] — первый токен (как он распределил внимание)
        # .detach() — отсоединяем от графа вычислений (не нужен для просмотра)
        print(f"Голова {head_i}: {w.round(decimals=2).tolist()}")
        # Каждая голова даёт разное распределение весов — они учатся разным аспектам

    from masks import make_causal_mask
    # Импортируем функцию из нашего файла masks.py

    mask = make_causal_mask(seq_len)
    # Создаём causal mask для seq_len=10: нижнетреугольная матрица

    out_masked = mha(x, x, x, mask=mask)
    # Запускаем с маской — токены не видят будущее
    print(f"\nС causal mask — выход: {out_masked.shape}")
    # Форма та же (2, 10, 512), но содержимое другое — будущее заблокировано

    total = sum(p.numel() for p in mha.parameters())
    # mha.parameters() — итерирует по всем обучаемым параметрам модели (W_Q, W_K, W_V, W_O + biases)
    # p.numel() — количество элементов в тензоре параметра
    # sum(...) — суммируем все

    print(f"\nПараметров в MHA: {total:,}")
    # :, — форматирование с разделителем тысяч (1,048,576 вместо 1048576)

    print(f"  W_Q: {mha.W_Q.weight.shape} = {mha.W_Q.weight.numel():,}")
    # W_Q.weight: (512, 512) = 262,144 параметра
    # Плюс W_Q.bias: (512,) = 512 параметров
    # Итого в W_Q: 262,656

    print(f"  W_K: {mha.W_K.weight.shape} = {mha.W_K.weight.numel():,}")
    print(f"  W_V: {mha.W_V.weight.shape} = {mha.W_V.weight.numel():,}")
    print(f"  W_O: {mha.W_O.weight.shape} = {mha.W_O.weight.numel():,}")
    # Каждая матрица 512×512 = 262,144 параметра
    # Итого только веса: 4 × 262,144 = 1,048,576 ≈ 1M параметров
