import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    # Слой позиционного кодирования — добавляет к эмбеддингам информацию о позиции токена
    # Это НЕ обучаемый слой — веса фиксированы формулой, не меняются при обучении

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        # d_model  — размерность эмбеддинга (должна совпадать с моделью, например 512)
        # max_len  — максимальная длина последовательности которую мы поддерживаем
        #            5000 с запасом — реальные предложения обычно короче
        # dropout  — вероятность обнуления случайных нейронов (регуляризация, 0.1 = 10%)

        super().__init__()
        # Обязательный вызов конструктора nn.Module

        self.dropout = nn.Dropout(p=dropout)
        # Dropout применяется после добавления позиционного кодирования
        # Помогает не переобучиться — случайно "выключает" 10% нейронов во время обучения
        # При inference (.eval()) dropout автоматически отключается

        # Создаём матрицу позиционного кодирования: (max_len, d_model)
        # Каждая строка = вектор для одной позиции
        pe = torch.zeros(max_len, d_model)
        # torch.zeros — матрица из нулей, потом заполним по формуле

        position = torch.arange(0, max_len).unsqueeze(1)
        # torch.arange(0, max_len) — числа от 0 до max_len-1: [0, 1, 2, ..., 4999]
        # .unsqueeze(1) — превращаем вектор (max_len,) в столбец (max_len, 1)
        # Это значения pos в формуле: PE(pos, i) = sin(pos / 10000^(2i/d_model))

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        # Вычисляем знаменатель 10000^(2i/d_model) для каждого i — но через exp/log
        # Почему не напрямую? Числовая стабильность: 10000^(512/512) = 10000 — большое число
        # Через логарифм: exp(2i * (-log(10000)/d_model)) = 10000^(-2i/d_model) = 1/10000^(2i/d_model)
        #
        # torch.arange(0, d_model, 2) — чётные индексы: [0, 2, 4, ..., d_model-2]
        # Нас интересует только d_model/2 значений (для sin и cos поровну)
        # div_term форма: (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 0::2] — все строки, чётные столбцы (0, 2, 4, ...)
        # position * div_term — broadcasting: (max_len,1) * (d_model/2,) = (max_len, d_model/2)
        # Заполняем чётные измерения синусом

        pe[:, 1::2] = torch.cos(position * div_term)
        # pe[:, 1::2] — все строки, нечётные столбцы (1, 3, 5, ...)
        # Заполняем нечётные измерения косинусом

        pe = pe.unsqueeze(0)
        # pe было (max_len, d_model), добавляем измерение батча → (1, max_len, d_model)
        # Единица на позиции батча: PyTorch растянет на любой размер батча (broadcasting)

        self.register_buffer('pe', pe)
        # register_buffer — регистрируем pe как "буфер" модуля, не как параметр
        # Разница от nn.Parameter:
        #   - parameter: обучается (обновляется при backward)
        #   - buffer: НЕ обучается, но сохраняется в state_dict и переносится с моделью на GPU
        # Нам нужен buffer: хотим чтобы pe переносился на GPU вместе с моделью (.to(device))

    def forward(self, x):
        # x: (batch, seq_len, d_model) — эмбеддинги токенов
        # Добавляем позиционное кодирование к каждому токену

        x = x + self.pe[:, :x.size(1)]
        # self.pe: (1, max_len, d_model)
        # self.pe[:, :x.size(1)] — берём только нужные позиции (0 до seq_len-1)
        #   x.size(1) = seq_len текущей последовательности (может быть короче max_len)
        # + — поэлементное сложение; broadcasting растянет батч автоматически
        # Каждый токен получает: его эмбеддинг + вектор его позиции

        return self.dropout(x)
        # Применяем dropout для регуляризации


if __name__ == "__main__":
    torch.manual_seed(42)

    d_model = 512   # размерность как в оригинальной статье
    max_len = 100   # для теста возьмём поменьше
    batch   = 2
    seq_len = 20    # предложения из 20 токенов

    pe_layer = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
    # dropout=0.0 чтобы не мешал при проверке чисел

    # Включаем режим eval — dropout не будет обнулять нейроны
    pe_layer.eval()

    x = torch.zeros(batch, seq_len, d_model)
    # Начинаем с нулевых эмбеддингов чтобы увидеть чистое позиционное кодирование

    out = pe_layer(x)
    # out = 0 + PE = просто позиционное кодирование

    print(f"Вход:  {x.shape}")
    print(f"Выход: {out.shape}")   # форма не меняется: (2, 20, 512)
    print(f"\nПервые 8 чисел позиционного вектора для позиций 0, 1, 2:")
    for pos in [0, 1, 2]:
        print(f"  pos={pos}: {out[0, pos, :8].tolist()}")
    # Видим что каждая позиция имеет уникальный вектор

    # ── Проверяем ключевое свойство: позиции различимы ──
    print("\n--- Различимость позиций ---")
    v0 = out[0, 0]   # вектор позиции 0
    v1 = out[0, 1]   # вектор позиции 1
    v5 = out[0, 5]   # вектор позиции 5

    cos = nn.CosineSimilarity(dim=0)
    # CosineSimilarity — мера схожести двух векторов, от -1 до 1
    # 1.0 = одинаковые направления, 0.0 = перпендикулярны, -1.0 = противоположны

    print(f"Схожесть pos=0 и pos=1: {cos(v0, v1):.4f}")  # близкие позиции → высокая схожесть
    print(f"Схожесть pos=0 и pos=5: {cos(v0, v5):.4f}")  # далёкие позиции → ниже
    # Это важно: модель может определять относительное расстояние между токенами

    # ── Смотрим паттерны разных измерений ──
    print("\n--- Паттерн: dim=0 (высокая частота) vs dim=100 (низкая частота) ---")
    print("dim=0  (sin, высокая частота):", [round(out[0, p, 0].item(), 3) for p in range(10)])
    print("dim=100 (sin, низкая частота):", [round(out[0, p, 100].item(), 3) for p in range(10)])
    # dim=0: быстро меняется (много колебаний)
    # dim=100: медленно меняется (почти одинаковый для соседних позиций)
    # Комбинация быстрых и медленных → уникальный паттерн для каждой позиции

    # ── Проверяем перенос на GPU если доступен ──
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pe_layer = pe_layer.to(device)
    x = x.to(device)
    out = pe_layer(x)
    print(f"\nУстройство: {device}")
    print(f"Выход на {device}: {out.shape}, device={out.device}")
    # register_buffer гарантирует что pe переехал вместе с моделью на device
