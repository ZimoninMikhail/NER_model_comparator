"""Графики, HTML-отчёты и подсветка результатов NER.

Исходник восстановлен по интерфейсам и оформлению сохранённой версии
от 7 мая 2026 года. Поддерживаются также методы поздней сокращённой версии.
"""

from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
import os
import webbrowser

import matplotlib.pyplot as plt


class NERVisualizer:
    COLORS = {
        'COUNTRY': '#FF6B6B', 'CITY': '#4ECDC4', 'LOCATION': '#45B7D1',
        'STATE_OR_PROV': '#96CEB4', 'DISTRICT': '#FFEAA7',
    }
    ENTITY_NAMES = {
        'COUNTRY': 'Страна', 'CITY': 'Город', 'LOCATION': 'Местоположение',
        'STATE_OR_PROV': 'Регион', 'DISTRICT': 'Район',
    }

    def __init__(self, output_dir='reports/visualization'):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'plots'
        self.html_dir = self.output_dir / 'html'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.html_dir.mkdir(parents=True, exist_ok=True)
        self.ANSI_COLORS = {
            'COUNTRY': '\033[91m', 'CITY': '\033[96m', 'LOCATION': '\033[94m',
            'STATE_OR_PROV': '\033[92m', 'DISTRICT': '\033[93m',
        }
        self.RESET = '\033[0m'

    def _get_timestamp(self):
        return datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    def _open_file(self, file_path):
        webbrowser.open(Path(file_path).resolve().as_uri())

    def _get_entity_color(self, entity_type):
        return self.COLORS.get(entity_type, '#DDDDDD')

    def _get_entity_name_ru(self, entity_type):
        return self.ENTITY_NAMES.get(entity_type, entity_type)

    def _save_plot(self, fig, name, save=True, open_result=False):
        path = None
        try:
            if save:
                path = self.plots_dir / f'{name}_{self._get_timestamp()}.png'
                fig.savefig(path, dpi=150, bbox_inches='tight')
                print(f'[SAVED] {path}')
            if open_result:
                if path is not None:
                    self._open_file(path)
                else:
                    plt.show()
        finally:
            plt.close(fig)
        return path

    def _save_html(self, content, name, save=True, open_result=False):
        if not save:
            return content
        path = self.html_dir / f'{name}_{self._get_timestamp()}.html'
        path.write_text(content, encoding='utf-8')
        print(f'[SAVED] {path}')
        if open_result:
            self._open_file(path)
        return path

    @staticmethod
    def _label_bars(ax, bars, percent=False, rotation=0, fontsize=9):
        for bar in bars:
            value = bar.get_height()
            label = f'{value:.1f}%' if percent else f'{value:g}'
            ax.annotate(label, (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=fontsize, rotation=rotation)

    def plot_entity_distribution(self, entities, title='Распределение типов сущностей',
                                 save=True, open_result=False):
        counts = dict(entities) if isinstance(entities, dict) else Counter(
            entity.get('type', 'UNKNOWN') for entity in entities)
        if not counts:
            return None
        items = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
        types, values = zip(*items)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(types, values, color=[self._get_entity_color(t) for t in types],
                      edgecolor='black')
        self._label_bars(ax, bars)
        ax.set(xlabel='Тип сущности', ylabel='Количество', title=title)
        ax.tick_params(axis='x', rotation=45)
        ax.margins(y=0.15)
        fig.tight_layout()
        return self._save_plot(fig, 'entity_distribution', save, open_result)

    def plot_metrics(self, metrics, title='Метрики модели', save=True, open_result=False):
        names = ['Precision', 'Recall', 'F1']
        values = [metrics.get(key, 0) * 100 for key in ['precision', 'recall', 'f1']]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(names, values, color=['#4ECDC4', '#96CEB4', '#FF6B6B'],
                      edgecolor='black')
        self._label_bars(ax, bars, percent=True)
        ax.set(ylim=(0, 105), ylabel='Проценты (%)', title=title)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        return self._save_plot(fig, 'metrics', save, open_result)

    def plot_loss_curve(self, losses, steps=None, title='Кривая обучения',
                        save=True, open_result=False):
        if not len(losses):
            return None
        if steps is None:
            steps = list(range(1, len(losses) + 1))
        if len(steps) != len(losses):
            raise ValueError('Количество шагов должно совпадать с количеством значений loss')
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, losses, 'o-', color='#4ECDC4', linewidth=2, markersize=4)
        ax.set(xlabel='Шаг', ylabel='Loss', title=title)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save_plot(fig, 'loss_curve', save, open_result)

    def plot_model_comparison(self, models_data, save=True, open_result=False):
        if not models_data:
            return None
        fig, ax = plt.subplots(figsize=(max(8, len(models_data)), 6))
        positions = list(range(len(models_data)))
        for offset, key, label, color in [
            (-0.25, 'precision', 'Precision', '#4ECDC4'),
            (0, 'recall', 'Recall', '#96CEB4'), (0.25, 'f1', 'F1', '#FF6B6B'),
        ]:
            bars = ax.bar([x + offset for x in positions],
                          [m.get(key, 0) * 100 for m in models_data],
                          width=0.25, label=label, color=color)
            self._label_bars(ax, bars, percent=True, rotation=90, fontsize=7)
        ax.set_xticks(positions)
        ax.set_xticklabels([m.get('version', str(i + 1)) for i, m in enumerate(models_data)],
                           rotation=45, ha='right', fontsize=9)
        ax.set(xlabel='Версия модели', ylabel='Значение (%)', ylim=(0, 105),
               title='Сравнение метрик моделей')
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        return self._save_plot(fig, 'model_comparison', save, open_result)

    def plot_benchmark(self, benchmark_data, save=True, open_result=False):
        times = benchmark_data.get('inference', {}).get('times_per_text', [])
        if not times:
            return None
        values = [t * 1000 for t in times]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(1, len(times) + 1), values, color='#4ECDC4', edgecolor='black')
        self._label_bars(ax, bars)
        mean = sum(values) / len(values)
        ax.axhline(mean, color='red', linestyle='--', label=f'Среднее: {mean:.1f} мс')
        ax.set(xlabel='Номер текста', ylabel='Время инференса (мс)',
               title='Производительность инференса')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.margins(y=0.15)
        fig.tight_layout()
        return self._save_plot(fig, 'benchmark', save, open_result)

    def create_dashboard(self, components, title='NER Dashboard', save=True, open_result=False):
        sections = []
        for component in components:
            heading = escape(str(component.get('title', '')))
            description = escape(str(component.get('description', '')))
            if component.get('type') == 'plot':
                path = Path(component['path']).resolve()
                if not path.is_file():
                    raise FileNotFoundError(f'График не найден: {path}')
                relative = Path(os.path.relpath(path, self.html_dir.resolve())).as_posix()
                content = f'<img src="{escape(relative, quote=True)}" alt="{heading}">'
            elif component.get('type') == 'html':
                content = component.get('content', '')
            else:
                continue
            sections.append(f'<section><h3>{heading}</h3><p>{description}</p>{content}</section>')
        safe_title = escape(title)
        content = ('<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8">'
                   f'<title>{safe_title}</title><style>body{{font-family:sans-serif;'
                   'max-width:1200px;margin:30px auto;padding:0 20px;}img{max-width:100%;}'
                   'section{margin:24px 0;} .ner-text{white-space:pre-wrap;line-height:1.8;}'
                   '</style></head><body>'
                   f'<h1>{safe_title}</h1><p>Generated: {datetime.now().isoformat()}</p>'
                   + ''.join(sections) + '</body></html>')
        return self._save_html(content, 'dashboard', save, open_result)

    def get_color_for_type(self, entity_type):
        return self.ANSI_COLORS.get(entity_type, self.RESET)

    def print_legend(self):
        print('\nЛегенда типов сущностей:')
        for entity_type, color in self.ANSI_COLORS.items():
            print(f'  {color}{entity_type}{self.RESET}')

    @staticmethod
    def _non_overlapping_entities(text, entities):
        cursor = 0
        for entity in sorted(entities, key=lambda e: (int(e.get('start', 0)),
                                                     -int(e.get('end', 0)))):
            start, end = int(entity.get('start', 0)), int(entity.get('end', 0))
            if 0 <= start < end <= len(text) and start >= cursor:
                yield entity, start, end
                cursor = end

    def highlight_entities_terminal(self, text, entities, show_legend=False):
        if show_legend:
            self.print_legend()
        chunks, cursor = [], 0
        for entity, start, end in self._non_overlapping_entities(text, entities):
            chunks.extend([text[cursor:start], self.get_color_for_type(entity.get('type')),
                           text[start:end], self.RESET])
            cursor = end
        chunks.append(text[cursor:])
        return ''.join(chunks)

    def highlight_entities_html(self, text, entities):
        chunks, cursor = [], 0
        for entity, start, end in self._non_overlapping_entities(text, entities):
            entity_type = entity.get('type', 'UNKNOWN')
            color = self._get_entity_color(entity_type)
            name = escape(str(self._get_entity_name_ru(entity_type)), quote=True)
            chunks.append(escape(text[cursor:start]))
            chunks.append(f'<span title="{name}" style="background-color:{color};'
                          'padding:2px 4px;border-radius:4px;">'
                          f'{escape(text[start:end])}</span>')
            cursor = end
        chunks.append(escape(text[cursor:]))
        return '<div class="ner-text">' + ''.join(chunks) + '</div>'

    def plot_consistency_over_batches(self, batch_consistencies,
                                      title='Динамика консистентности по батчам',
                                      save=True, open_result=False):
        if not batch_consistencies:
            return None
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot([c.get('batch_num', i + 1) for i, c in enumerate(batch_consistencies)],
                [c.get('consistency_ratio', 0) for c in batch_consistencies], 'o-')
        ax.set(xlabel='Номер батча', ylabel='Доля консистентных сущностей',
               ylim=(0, 1.05), title=title)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save_plot(fig, 'consistency_over_batches', save, open_result)

    def plot_filter_impact(self, before_counts, after_counts,
                          title='Распределение сущностей до и после фильтрации',
                          save=True, open_result=False):
        types = sorted(set(before_counts) | set(after_counts))
        if not types:
            return None
        fig, ax = plt.subplots(figsize=(14, 7))
        positions = list(range(len(types)))
        ax.bar([x - 0.175 for x in positions], [before_counts.get(t, 0) for t in types],
               width=0.35, label='До фильтрации')
        ax.bar([x + 0.175 for x in positions], [after_counts.get(t, 0) for t in types],
               width=0.35, label='После фильтрации')
        ax.set_xticks(positions)
        ax.set_xticklabels(types, rotation=45, ha='right')
        ax.set(xlabel='Тип сущности', ylabel='Количество', title=title)
        ax.legend()
        fig.tight_layout()
        return self._save_plot(fig, 'filter_impact', save, open_result)

    def plot_length_distribution(self, lengths, title='Распределение длины документов',
                                 save=True, open_result=False):
        if not len(lengths):
            return None
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(lengths, bins=30, color='#4ECDC4', edgecolor='black')
        ax.set(xlabel='Длина документа (символы)', ylabel='Частота', title=title)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save_plot(fig, 'length_distribution', save, open_result)
