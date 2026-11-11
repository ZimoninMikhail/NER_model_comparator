"""Повторные замеры инференса; память измеряется отдельным проходом."""
import hashlib
import json
import platform
import statistics
import threading
import time

import numpy as np
import psutil
import torch


def _sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def measure_inference(pipeline, texts, warmup_runs=2, repeats=5, rss_interval_sec=0.005):
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError('Для бенчмарка нужен непустой набор непустых текстов')
    if type(warmup_runs) is not int or warmup_runs < 0 or type(repeats) is not int or repeats <= 0:
        raise ValueError('warmup_runs должен быть >= 0, repeats должен быть > 0')
    if not 0 < rss_interval_sec <= 1:
        raise ValueError('Недопустимый интервал измерения RSS')
    device = pipeline.device
    for _ in range(warmup_runs):
        for text in texts:
            pipeline.predict(text)
    _sync(device)
    times = [[] for _ in texts]
    entity_counts = [0] * len(texts)
    for _ in range(repeats):
        for index, text in enumerate(texts):
            _sync(device)
            started = time.perf_counter()
            entities = pipeline.predict(text)
            _sync(device)
            elapsed = time.perf_counter() - started
            if elapsed <= 0:
                raise ValueError('Таймер вернул неположительную длительность')
            times[index].append(elapsed)
            entity_counts[index] = len(entities)
    # Сэмплер RSS не влияет на измеряемые выше задержки.
    process = psutil.Process()
    baseline = process.memory_info().rss
    samples = [baseline]
    stop = threading.Event()
    def sample_rss():
        while not stop.wait(rss_interval_sec):
            samples.append(process.memory_info().rss)
    memory = {'method': 'separate_pass_after_warmup',
              'rss_sampling_interval_sec': rss_interval_sec}
    if device == 'cuda':
        _sync(device)
        cuda_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    try:
        for text in texts:
            result = pipeline.predict(text)
            _sync(device)
            samples.append(process.memory_info().rss)
            del result
        if device == 'cuda':
            memory['gpu'] = {
                'baseline_allocated_mb': cuda_before / 1024**2,
                'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2,
                'peak_allocated_delta_mb': max(0, torch.cuda.max_memory_allocated() - cuda_before) / 1024**2,
                'scope': 'pytorch_allocator_current_process'}
    finally:
        stop.set()
        sampler.join()
    memory.update(cpu_rss_baseline_mb=baseline / 1024**2,
        cpu_rss_sampled_peak_mb=max(samples) / 1024**2,
        cpu_rss_sampled_peak_delta_mb=max(0, max(samples) - baseline) / 1024**2,
        rss_samples=len(samples),
        note='RSS — сэмплированная память всего процесса, включая модель; короткие пики могут быть пропущены. '
             'CUDA — память аллокатора PyTorch, не всей видеокарты.')
    all_times = [value for row in times for value in row]
    return {'architecture': getattr(pipeline, 'architecture', 'bert'),
        'model_name': pipeline.model_path.name, 'model_path': str(pipeline.model_path),
        'device': device, 'num_texts': len(texts), 'num_measurements': len(all_times),
        'warmup_runs': warmup_runs, 'repeats': repeats,
        'input_sha256': hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest(),
        'environment': {'python': platform.python_version(), 'pytorch': str(torch.__version__),
                        'platform': platform.platform()},
        'inference': {'avg_time_sec': statistics.mean(all_times),
            'avg_time_ms': statistics.mean(all_times) * 1000,
            'median_time_ms': statistics.median(all_times) * 1000,
            'p95_time_ms': float(np.percentile(all_times,95)) * 1000,
            'min_time_sec': min(all_times), 'max_time_sec': max(all_times),
            'std_time_sec': statistics.pstdev(all_times),
            'times_per_text': [statistics.mean(row) for row in times],
            'measurements_sec': times},
        'throughput': len(all_times) / sum(all_times), 'memory': memory,
        'predictions': [{'text': text[:100], 'num_entities': count}
                        for text,count in zip(texts,entity_counts)]}
