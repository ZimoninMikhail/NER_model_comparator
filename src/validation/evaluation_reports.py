"""Чтение законченных оценок и группировка сравнимых результатов."""
import json
from pathlib import Path


def load_evaluation_reports(reports_dir):
    reports = []
    for path in sorted((Path(reports_dir)/'evaluation').glob('*/report.json')):
        try:
            report = json.loads(path.read_text(encoding='utf-8'))
        except (ValueError,OSError) as error:
            raise ValueError(f'Не удалось прочитать отчёт оценки: {path}') from error
        if report.get('status') != 'complete':
            continue
        if (report.get('format_version') != 1 or report.get('architecture') not in ('bert','bilstm_crf','gliner')
                or report.get('split') not in ('dev','test') or not report.get('dataset_version')
                or 'macro_f1' not in report.get('metrics',{})):
            raise ValueError(f'Неподдерживаемый отчёт оценки: {path}')
        reports.append(report)
    return sorted(reports,key=lambda report:report.get('timestamp',''))


def comparison_groups(model_entries, evaluation_reports):
    """Dev/test, версии, схемы и разные объёмы оценки не смешиваются."""
    groups = {}
    def add(identifier, architecture, metrics, metadata, timestamp):
        if 'macro_f1' not in metrics or 'micro_f1' not in metrics:
            return
        dataset = metadata.get('dataset_version')
        split = metadata.get('split','dev')
        types = tuple(sorted(metrics.get('entity_types',[])))
        protocol = (dataset,split,types,metrics.get('match'),
                    metadata.get('preprocessing_version'),metrics.get('num_documents'))
        # Для старой записи без происхождения не заявляется сравнимость.
        if not dataset or not types:
            protocol = protocol + (identifier,)
        label = {'bert':'BERT','bilstm_crf':'BiLSTM-CRF','gliner':'GLiNER (без дообучения)'}.get(architecture,architecture)
        group = groups.setdefault(protocol,{'dataset_version':dataset,'split':split,'models':[]})
        group['models'].append({'display_name':label+'\n'+identifier,'version':identifier,
            'macro_f1':metrics['macro_f1'],'micro_f1':metrics['micro_f1'],'timestamp':timestamp})
    for entry in model_entries:
        metadata=entry.get('metadata',{})
        add(entry['version'],metadata.get('architecture','bert'),entry.get('metrics',{}),metadata,entry.get('timestamp',''))
    for report in evaluation_reports:
        add(report['evaluation_id'],report['architecture'],report['metrics'],report,report.get('timestamp',''))
    for group in groups.values():
        group['models'].sort(key=lambda row:row['timestamp'])
    return list(groups.values())
