import torch                    # работа с тензорами
import torch.nn.functional as F  # функции: softmax и др.
import math                      # нужен для math.sqrt в attention_with_mask


def make_padding_mask(seq, pad_idx=0):
    # Создаёт маску которая помечает где реальные токены, а где PAD
    #
    # seq: (batch, seq_len) — тензор с индексами токенов
    #   Например: [[5, 3, 8, 0, 0], [2, 7, 0, 0, 0]]
    #   0 — это PAD (заглушка, чтобы выровнять длину)
    #
    # pad_idx=0 — индекс PAD-токена (по умолчанию 0)

    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    # (seq != pad_idx) — сравниваем каждый элемент с pad_idx
    #   Получаем булев тензор (batch, seq_len): True где реальный токен, False где PAD
    #   Пример: [[T, T, T, F, F], [T, T, F, F, F]]
    #
    # .unsqueeze(1) — добавляем измерение на позицию 1
    #   (batch, seq_len) → (batch, 1, seq_len)
    #   Зачем: нужно чтобы маска "растягивалась" на все головы attention
    #
    # .unsqueeze(2) — добавляем ещё одно измерение
    #   (batch, 1, seq_len) → (batch, 1, 1, seq_len)
    #   Зачем: форма (batch, 1, 1, seq_len) совместима с scores (batch, heads, seq_q, seq_k)
    #   PyTorch автоматически "растянет" маску на все головы и все строки (broadcasting)


def make_causal_mask(seq_len):
    # Создаёт маску которая запрещает смотреть на будущие токены
    # Нужна в decoder: при обучении нельзя подсматривать ответ вперёд

    mask = torch.tril(torch.ones(seq_len, seq_len))
    # torch.ones(seq_len, seq_len) — матрица из единиц (seq_len × seq_len)
    #   Пример для seq_len=4:
    #   [[1, 1, 1, 1],
    #    [1, 1, 1, 1],
    #    [1, 1, 1, 1],
    #    [1, 1, 1, 1]]
    #
    # torch.tril(...) — оставляет только нижний треугольник (lower triangle)
    #   [[1, 0, 0, 0],
    #    [1, 1, 0, 0],
    #    [1, 1, 1, 0],
    #    [1, 1, 1, 1]]
    #   Строка i = токен i; 1 значит "можно смотреть", 0 значит "нельзя (будущее)"

    return mask.unsqueeze(0).unsqueeze(0)
    # Добавляем два измерения в начало чтобы форма совпала с scores
    # (seq_len, seq_len) → (1, 1, seq_len, seq_len)
    # Единицы на первых позициях: PyTorch растянет на batch и heads автоматически


def make_decoder_mask(tgt_seq, pad_idx=0):
    # Decoder нуждается в ДВУХ масках одновременно:
    # 1. Не смотреть на PAD-токены (padding mask)
    # 2. Не смотреть на будущие токены (causal mask)
    # Объединяем их через AND: токен разрешён только если ОБА условия выполнены

    seq_len = tgt_seq.size(1)
    # tgt_seq.size(1) — длина последовательности (второе измерение тензора)
    # Нужно чтобы создать causal mask правильного размера

    pad_mask = make_padding_mask(tgt_seq, pad_idx)
    # Получаем padding mask: (batch, 1, 1, seq_len)
    # True где реальный токен, False где PAD

    causal = make_causal_mask(seq_len).to(tgt_seq.device)
    # Получаем causal mask: (1, 1, seq_len, seq_len)
    # .to(tgt_seq.device) — переносим маску на то же устройство что и данные
    # (CPU или GPU — они должны быть на одном устройстве для операций)

    return pad_mask & causal.bool()
    # causal.bool() — конвертируем float маску (0.0/1.0) в булеву (False/True)
    # & — логическое AND: True только если оба True
    # Broadcasting: (batch, 1, 1, seq_len) & (1, 1, seq_len, seq_len)
    #   → (batch, 1, seq_len, seq_len)
    # Результат: токен разрешён только если он не PAD И не в будущем


def attention_with_mask(Q, K, V, mask=None):
    # Упрощённая версия attention — только для демонстрации масок
    # (без сохранения весов, без возврата весов отдельно)

    d_k = Q.size(-1)
    # Берём размерность последнего измерения — это d_k

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # Считаем сырые оценки схожести Q с K, нормализуем на √d_k

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
        # Там где маска=0 (нельзя смотреть) ставим -inf
        # softmax(-inf) = 0, значит эти позиции не влияют на результат

    return F.softmax(scores, dim=-1)
    # Возвращаем только веса (не вычисляем выход × V)
    # Нам здесь важно показать как маска меняет веса


if __name__ == "__main__":

    print("=" * 50)
    print("PADDING MASK")
    print("=" * 50)

    batch_tokens = torch.tensor([
        [5, 3, 8, 0, 0],  # предложение 1: токены 5,3,8 + два PAD (0)
        [2, 7, 0, 0, 0],  # предложение 2: токены 2,7 + три PAD
        [1, 4, 6, 9, 3],  # предложение 3: пять реальных токенов, PAD нет
    ])
    # Это батч из 3 предложений
    # Все три выровнены до длины 5 (самое длинное)
    # 0 — это PAD, просто заглушка

    pad_mask = make_padding_mask(batch_tokens, pad_idx=0)
    # Создаём маску: True где токен ≠ 0

    print(f"Токены:\n{batch_tokens}")
    print(f"\nPadding mask (1=смотрим, 0=игнорируем):")
    print(pad_mask.squeeze())
    # .squeeze() убирает все измерения размером 1 — для красивого вывода
    # Было (3, 1, 1, 5) → стало (3, 5)

    torch.manual_seed(0)
    # Фиксируем seed для воспроизводимости

    seq_len, d_k = 5, 8
    # seq_len=5 — длина последовательности
    # d_k=8 — маленькая размерность для наглядности

    Q = torch.randn(1, 1, seq_len, d_k)
    # (1 батч, 1 голова, 5 токенов, 8 чисел)
    # Имитируем Query для одного примера

    K = torch.randn(1, 1, seq_len, d_k)
    # Аналогично для Key

    V = torch.randn(1, 1, seq_len, d_k)
    # Аналогично для Value (здесь не используется — нас интересуют только веса)

    mask_single = make_padding_mask(batch_tokens[0:1], pad_idx=0)
    # batch_tokens[0:1] — берём только первое предложение [5, 3, 8, 0, 0]
    # [0:1] вместо [0] — чтобы сохранить измерение батча (форма (1,5) а не (5,))
    # Создаём маску для него: [True, True, True, False, False]

    weights_no_mask = attention_with_mask(Q, K, V, mask=None)
    # Считаем веса БЕЗ маски — все 5 позиций участвуют

    weights_masked = attention_with_mask(Q, K, V, mask=mask_single)
    # Считаем веса С маской — позиции 3 и 4 (PAD) игнорируются

    print(f"\nВеса БЕЗ маски (строка 0):")
    print(weights_no_mask[0, 0, 0].detach().round(decimals=3))
    # [0, 0, 0] — первый батч, первая голова, первый токен (Q[0])
    # Смотрим как токен 0 распределяет внимание по всем 5 позициям

    print(f"\nВеса С padding mask (позиции 3,4 должны быть 0):")
    print(weights_masked[0, 0, 0].detach().round(decimals=3))
    # Позиции 3 и 4 должны быть 0.0 — PAD игнорируется
    # Оставшиеся веса (позиции 0,1,2) перераспределились и дают сумму 1.0

    print("\n" + "=" * 50)
    print("CAUSAL MASK")
    print("=" * 50)

    seq_len = 5
    causal = make_causal_mask(seq_len)
    # Создаём causal mask для последовательности длиной 5

    print(f"\nCausal mask для seq_len={seq_len}:")
    print(causal.squeeze())
    # Нижнетреугольная матрица 5×5
    # Строка 0 (токен 0): видит только позицию 0
    # Строка 1 (токен 1): видит позиции 0 и 1
    # Строка 4 (токен 4): видит все позиции 0-4

    print("\nЧитать так: строка i = токен i, столбец j = на кого смотрит")
    print("1 = можно смотреть, 0 = нельзя (будущее)")

    print("\n" + "=" * 50)
    print("DECODER MASK (PAD + CAUSAL)")
    print("=" * 50)

    tgt = torch.tensor([[5, 3, 8, 0, 0]])
    # Одно предложение: токены 5, 3, 8 и два PAD в конце
    # [[...]] — двойные скобки чтобы форма была (1, 5), а не (5,)
    # Форма с батчем нужна для make_decoder_mask

    dec_mask = make_decoder_mask(tgt, pad_idx=0)
    # Объединяем padding + causal в одну маску

    print(f"\nTarget токены: {tgt.tolist()}")
    print(f"\nDecoder mask:")
    print(dec_mask.squeeze().int())
    # .int() — конвертируем True/False в 1/0 для наглядности

    print("\nТокен 0 ('5') видит только себя")
    # Строка 0: [1,0,0,0,0] — causal запрещает всё кроме позиции 0

    print("Токен 2 ('8') видит 0,1,2 — но НЕ PAD позиции 3,4")
    # Строка 2: [1,1,1,0,0] — causal разрешает 0,1,2; padding блокирует 3,4

    print("Токен 3 (PAD) никуда не смотрит из-за causal + всё равно PAD")
    # Строка 3: [1,1,1,0,0] — causal разрешает 0,1,2,3; но 3 и 4 это PAD → 0
