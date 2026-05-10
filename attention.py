import torch                    # основная библиотека для работы с тензорами (многомерными массивами)
import torch.nn as nn            # модуль с готовыми слоями нейронных сетей (Linear, Dropout и т.д.)
import torch.nn.functional as F  # функции без состояния: softmax, relu и т.д. (без обучаемых весов)
import math                      # стандартная математика Python — нужен для math.sqrt


def scaled_dot_product_attention(Q, K, V, mask=None):
    # Функция реализует формулу: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
    #
    # Q (Query)  — "вопрос": что ищет текущий токен?         форма: (batch, heads, seq_len, d_k)
    # K (Key)    — "заголовок": что предлагает каждый токен? форма: (batch, heads, seq_len, d_k)
    # V (Value)  — "содержимое": что отдаёт каждый токен?    форма: (batch, heads, seq_len, d_k)
    # mask       — маска, где 0 = "сюда смотреть нельзя"
    #
    # batch  — сколько предложений обрабатывается одновременно
    # heads  — количество голов (параллельных механизмов внимания)
    # seq_len — длина последовательности (количество токенов)
    # d_k    — размерность вектора одной головы

    d_k = Q.size(-1)
    # Q.size(-1) — берём последнее измерение тензора Q, то есть d_k
    # Нужно для деления на sqrt(d_k) — чтобы дисперсия оставалась ≈1

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # torch.matmul(Q, K.transpose(-2, -1)) — матричное умножение Q на транспонированный K
    # K.transpose(-2, -1) меняет местами два последних измерения:
    #   K было (batch, heads, seq_len, d_k)
    #   стало  (batch, heads, d_k, seq_len)
    # После умножения Q @ K^T получаем (batch, heads, seq_len, seq_len)
    # Каждая ячейка [b, h, i, j] = "насколько токен i похож на токен j в голове h"
    # Делим на sqrt(d_k) чтобы не было насыщения softmax (объяснено в why_sqrt_dk.py)

    if mask is not None:
        # Если маска передана — применяем её
        scores = scores.masked_fill(mask == 0, float('-inf'))
        # masked_fill заменяет значения в scores на -inf там где mask == 0
        # Почему -inf? Потому что softmax(−inf) = 0, то есть токен получит вес 0
        # и не будет влиять на результат — мы его полностью игнорируем

    weights = F.softmax(scores, dim=-1)
    # softmax превращает сырые оценки (scores) в вероятности (веса)
    # dim=-1 означает "нормализуем вдоль последнего измерения" — то есть по столбцам
    # Для каждого запроса i сумма весов по всем ключам j равна 1.0
    # weights форма: (batch, heads, seq_len, seq_len)

    output = torch.matmul(weights, V)
    # Умножаем веса на значения V: каждый токен получает взвешенную сумму всех V
    # weights: (batch, heads, seq_len, seq_len)
    # V:       (batch, heads, seq_len, d_k)
    # output:  (batch, heads, seq_len, d_k)
    # Токен i получает: sum_j(weight[i,j] * V[j]) — то есть "смесь" всех токенов

    return output, weights
    # Возвращаем и результат и веса — веса полезны для визуализации и отладки


if __name__ == "__main__":
    # Этот блок запускается только если файл запущен напрямую (python attention.py)
    # Если файл импортируется в другой — этот блок не выполняется

    batch, heads, seq_len, d_k = 2, 4, 10, 64
    # batch=2   — обрабатываем 2 предложения одновременно
    # heads=4   — 4 параллельные головы внимания
    # seq_len=10 — каждое предложение из 10 токенов
    # d_k=64    — каждый токен представлен вектором из 64 чисел

    Q = torch.randn(batch, heads, seq_len, d_k)
    # torch.randn заполняет тензор случайными числами из нормального распределения N(0,1)
    # В реальности Q получается из входа через обучаемую матрицу W_Q
    # Здесь просто имитируем чтобы проверить что функция работает

    K = torch.randn(batch, heads, seq_len, d_k)
    # Аналогично для ключей — в реальности это W_K * вход

    V = torch.randn(batch, heads, seq_len, d_k)
    # Аналогично для значений — в реальности это W_V * вход

    out, weights = scaled_dot_product_attention(Q, K, V)
    # Запускаем attention без маски — все токены видят всех

    print(f"Q shape:       {Q.shape}")
    # Проверяем форму входа — должно быть (2, 4, 10, 64)

    print(f"Output shape:  {out.shape}")
    # Выход должен иметь ту же форму что и вход: (2, 4, 10, 64)
    # Каждый токен "обновил" своё представление на основе всех остальных

    print(f"Weights shape: {weights.shape}")
    # (2, 4, 10, 10) — для каждой пары (токен_i, токен_j) есть вес
    # Это матрица внимания: строка i = как токен i распределяет внимание по всем j

    print(f"Weights sum:   {weights[0, 0, 0].sum():.4f}")
    # weights[0, 0, 0] — первый батч, первая голова, первый токен
    # .sum() — сумма весов по всем ключам, должна быть 1.0 (свойство softmax)

    mask = torch.ones(batch, 1, seq_len, seq_len).tril()
    # torch.ones(...) — матрица из единиц формы (batch, 1, seq_len, seq_len)
    # .tril() — оставляет только нижний треугольник (triangle lower)
    # Результат: 1 на диагонали и ниже, 0 выше диагонали
    # Это causal mask: токен i может смотреть только на токены 0..i

    out_masked, w_masked = scaled_dot_product_attention(Q, K, V, mask)
    # Запускаем с маской — токены не видят будущее

    print(f"\nCausal mask — верхний треугольник весов (должен быть 0):")
    print(w_masked[0, 0].detach().round(decimals=3))
    # w_masked[0, 0] — первый батч, первая голова, матрица (seq_len, seq_len)
    # .detach() — отсоединяем от графа вычислений (не нужен для просмотра)
    # .round(decimals=3) — округляем до 3 знаков для читаемости
    # Верхний треугольник (будущее) должен быть 0.0
