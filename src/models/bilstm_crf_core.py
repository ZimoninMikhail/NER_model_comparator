"""Ядро BiLSTM-CRF на PyTorch; общая обёртка находится в соседнем bilstm_crf.py.

Метки и ID токенов совместимы с существующей подготовкой. Весов BERT здесь нет.
CRF получает только реальные текстовые токены: -100 никогда не является классом.
"""
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _mask(value: Optional[Tensor], shape, device, name):
    if value is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if value.shape != shape or value.device != device:
        raise ValueError(f'{name}: неверная форма или устройство')
    if not bool(((value == 0) | (value == 1)).all()):
        raise ValueError(f'{name}: допустимы только 0 и 1')
    return value.bool()


class LinearChainCRF(nn.Module):
    """CRF первого порядка, batch-first; transitions[предыдущая, следующая].

    forward возвращает отрицательный log-likelihood. mask должен быть префиксным;
    полностью пустые строки допустимы и не входят в знаменатель mean.
    При id2label запрещены начальный I и переходы в I другого/отсутствующего типа.
    """
    def __init__(self, num_tags: int, id2label: Optional[Mapping] = None):
        super().__init__()
        if type(num_tags) is not int or num_tags < 1:
            raise ValueError('num_tags должен быть положительным целым числом')
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        allowed_start = torch.ones(num_tags, dtype=torch.bool)
        allowed_transition = torch.ones(num_tags, num_tags, dtype=torch.bool)
        if id2label is not None:
            labels = {int(k): v for k, v in id2label.items()}
            if set(labels) != set(range(num_tags)) or len(set(labels.values())) != num_tags:
                raise ValueError('id2label должен однозначно описывать все BIO-метки')
            names = list(labels.values())
            if names.count('O') != 1 or any(
                name != 'O' and (not isinstance(name, str) or len(name) <= 2
                                or name[:2] not in ('B-', 'I-')) for name in names):
                raise ValueError('id2label должен содержать O и корректные B-/I-метки')
            if any(('I-' + name[2:]) not in names for name in names if name.startswith('B-')) or any(
                ('B-' + name[2:]) not in names for name in names if name.startswith('I-')):
                raise ValueError('Для каждого типа нужны обе BIO-метки: B и I')
            for current in range(num_tags):
                if labels[current].startswith('I-'):
                    allowed_start[current] = False
                    for previous in range(num_tags):
                        allowed_transition[previous, current] = labels[previous] in (
                            'B-' + labels[current][2:], 'I-' + labels[current][2:])
        self.register_buffer('allowed_start', allowed_start)
        self.register_buffer('allowed_transition', allowed_transition)
        for parameter in (self.start_transitions, self.end_transitions, self.transitions):
            nn.init.uniform_(parameter, -0.1, 0.1)

    def _validate(self, emissions, tags, mask):
        if emissions.ndim != 3 or emissions.shape[0] < 1 or emissions.shape[2] != self.num_tags:
            raise ValueError('emissions: ожидается [batch, tokens, num_tags] с непустым batch')
        if not emissions.is_floating_point() or not bool(torch.isfinite(emissions).all()):
            raise ValueError('emissions должны быть конечными вещественными числами')
        if emissions.device != self.transitions.device:
            raise ValueError('emissions и CRF должны находиться на одном устройстве')
        valid = _mask(mask, emissions.shape[:2], emissions.device, 'mask')
        if valid.shape[1] > 1 and bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError('mask CRF должен быть префиксным; сначала уплотните текстовые токены')
        if tags is not None:
            if tags.shape != valid.shape or tags.device != emissions.device or tags.dtype != torch.long:
                raise ValueError('tags: нужны целочисленные long-метки той же формы и устройства')
            if bool(((tags[valid] < 0) | (tags[valid] >= self.num_tags)).any()):
                raise ValueError('Неизвестная метка на реальном токене; -100 допустим только под маской')
            safe = tags.masked_fill(~valid, 0)
            if safe.shape[1]:
                if bool((valid[:, 0] & ~self.allowed_start[safe[:, 0]]).any()):
                    raise ValueError('Некорректный эталон BIO: последовательность начинается с I')
                if bool((valid[:, 1:] & ~self.allowed_transition[safe[:, :-1], safe[:, 1:]]).any()):
                    raise ValueError('Некорректный эталон BIO: запрещённый переход в I')
        return valid

    def _parameters_with_constraints(self):
        return (self.start_transitions.masked_fill(~self.allowed_start, float('-inf')),
                self.transitions.masked_fill(~self.allowed_transition, float('-inf')))

    def forward(self, emissions, tags, mask=None, reduction='mean'):
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError('reduction: допустимы none, sum, mean, token_mean')
        valid = self._validate(emissions, tags, mask)
        batch, steps, _ = emissions.shape
        if not steps:
            loss = emissions.sum(dim=(1, 2)) * 0 + (
                self.transitions.sum() + self.start_transitions.sum() + self.end_transitions.sum()) * 0
        else:
            start, transitions = self._parameters_with_constraints()
            safe = tags.masked_fill(~valid, 0)
            rows = torch.arange(batch, device=emissions.device)
            scores = start.unsqueeze(0) + emissions[:, 0]
            gold = torch.where(valid[:, 0], start[safe[:, 0]] + emissions[rows, 0, safe[:, 0]], 0.)
            for step in range(1, steps):
                candidates = scores.unsqueeze(2) + transitions.unsqueeze(0) + emissions[:, step].unsqueeze(1)
                scores = torch.where(valid[:, step, None], torch.logsumexp(candidates, dim=1), scores)
                contribution = transitions[safe[:, step - 1], safe[:, step]] + emissions[rows, step, safe[:, step]]
                gold = gold + torch.where(valid[:, step], contribution, 0.)
            lengths = valid.long().sum(dim=1)
            has_text = lengths > 0
            last = safe.gather(1, (lengths - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
            gold = gold + torch.where(has_text, self.end_transitions[last], 0.)
            partition = torch.where(has_text, torch.logsumexp(scores + self.end_transitions, dim=1), 0.)
            loss = partition - gold
        if reduction == 'none': return loss
        if reduction == 'sum': return loss.sum()
        denominator = valid.sum() if reduction == 'token_mean' else valid.any(dim=1).sum()
        return loss.sum() / denominator.clamp_min(1)

    @torch.no_grad()
    def decode(self, emissions, mask=None, ignore_index=-100):
        valid = self._validate(emissions, None, mask)
        batch, steps, _ = emissions.shape
        output = torch.full((batch, steps), ignore_index, dtype=torch.long, device=emissions.device)
        if not steps: return output
        start, transitions = self._parameters_with_constraints()
        scores = start.unsqueeze(0) + emissions[:, 0]
        history = []
        for step in range(1, steps):
            best_scores, previous = (scores.unsqueeze(2) + transitions.unsqueeze(0)).max(dim=1)
            scores = torch.where(valid[:, step, None], best_scores + emissions[:, step], scores)
            history.append(previous)
        best_last = (scores + self.end_transitions).argmax(dim=1)
        for row, length in enumerate(valid.sum(dim=1).tolist()):
            if length == 0: continue
            tag = best_last[row]
            output[row, length - 1] = tag
            for step in range(length - 2, -1, -1):
                tag = history[step][row, tag]
                output[row, step] = tag
        return output


@dataclass
class BiLSTMCRFOutput:
    loss: Optional[Tensor]
    emissions: Tensor
    predictions: Optional[Tensor]
    token_mask: Tensor


class BiLSTMCRF(nn.Module):
    """Embedding → двунаправленная LSTM → Linear → CRF.

    hidden_size — размер каждого направления. token_mask (по offsets != (0,0))
    имеет приоритет перед special_token_ids, позволяя отличать буквальный текст
    '[CLS]' от добавленного служебного токена. Без token_mask служебные ID нужно
    явно передать в конструктор. Маска никогда не выводится из эталонных labels.
    Все выдаваемые координаты последовательности соответствуют исходному входу;
    исключённые позиции имеют predictions=-100. Пустые тексты допустимы.
    """
    def __init__(self, vocab_size: int, num_labels: int, embedding_dim: int = 128,
                 hidden_size: int = 128, num_layers: int = 1, dropout: float = 0.2,
                 padding_idx: int = 0, special_token_ids: Sequence[int] = (),
                 id2label: Optional[Mapping] = None):
        super().__init__()
        for name, value in [('vocab_size', vocab_size), ('num_labels', num_labels),
                            ('embedding_dim', embedding_dim), ('hidden_size', hidden_size),
                            ('num_layers', num_layers)]:
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} должен быть положительным целым числом')
        if type(padding_idx) is not int or not 0 <= padding_idx < vocab_size:
            raise ValueError('Некорректный padding_idx')
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError('dropout должен лежать в [0, 1)')
        if any(type(token) is not int or not 0 <= token < vocab_size for token in special_token_ids):
            raise ValueError('Некорректные special_token_ids')
        self.padding_idx = padding_idx
        self.special_token_ids = tuple(sorted(set(special_token_ids)))
        self.num_labels = num_labels
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)
        self.crf = LinearChainCRF(num_labels, id2label)

    def forward(self, input_ids, attention_mask=None, labels=None, token_mask=None, decode=True):
        if input_ids.ndim != 2 or min(input_ids.shape) < 1 or input_ids.dtype != torch.long:
            raise ValueError('input_ids: ожидается непустой long-тензор [batch, tokens]')
        if input_ids.device != self.embedding.weight.device:
            raise ValueError('input_ids и модель должны находиться на одном устройстве')
        if bool(((input_ids < 0) | (input_ids >= self.embedding.num_embeddings)).any()):
            raise ValueError('ID токена вне словаря')
        attention = (input_ids != self.padding_idx) if attention_mask is None else _mask(
            attention_mask, input_ids.shape, input_ids.device, 'attention_mask')
        if token_mask is None:
            valid = attention.clone()
            for token in self.special_token_ids:
                valid &= input_ids != token
        else:
            valid = attention & _mask(token_mask, input_ids.shape, input_ids.device, 'token_mask')
        if labels is not None:
            if labels.shape != input_ids.shape or labels.dtype != torch.long or labels.device != input_ids.device:
                raise ValueError('labels: нужны long-метки той же формы и устройства, что input_ids')
            if bool(((labels[valid] < 0) | (labels[valid] >= self.num_labels)).any()):
                raise ValueError('Нет корректной метки реального токена; задайте token_mask для служебных позиций')
        lengths = valid.long().sum(dim=1)
        width = int(lengths.max().item())
        batch = input_ids.shape[0]
        rows, columns = valid.nonzero(as_tuple=True)
        ranks = valid.long().cumsum(dim=1)[rows, columns] - 1
        compact_ids = input_ids.new_full((batch, width), self.padding_idx)
        compact_ids[rows, ranks] = input_ids[rows, columns]
        compact_mask = torch.arange(width, device=input_ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        embeddings = self.dropout(self.embedding(compact_ids))
        encoded = embeddings.new_zeros((batch, width, self.lstm.hidden_size * 2))
        nonempty = (lengths > 0).nonzero(as_tuple=True)[0]
        if len(nonempty):
            packed = pack_padded_sequence(embeddings[nonempty], lengths[nonempty].cpu(),
                                          batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed)
            unpacked, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=width)
            encoded = encoded.index_copy(0, nonempty, unpacked)
        compact_emissions = self.classifier(self.dropout(encoded))
        emissions = compact_emissions.new_zeros((*input_ids.shape, self.num_labels))
        emissions[rows, columns] = compact_emissions[rows, ranks]
        loss = None
        if labels is not None:
            compact_tags = input_ids.new_full((batch, width), -100)
            compact_tags[rows, ranks] = labels[rows, columns]
            loss = self.crf(compact_emissions, compact_tags, compact_mask)
        predictions = None
        if decode:
            compact_predictions = self.crf.decode(compact_emissions, compact_mask)
            predictions = input_ids.new_full(input_ids.shape, -100)
            predictions[rows, columns] = compact_predictions[rows, ranks]
        return BiLSTMCRFOutput(loss, emissions, predictions, valid)
