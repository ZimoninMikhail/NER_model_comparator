"""GLiNER с общим выбором границ и глобальными координатами сущностей."""
from src.data_preparation.document_chunker import chunk_spans
from src.models.gliner import predict_entities, _input_dimensions, _validate_type_mapping


class GLiNERPredictor:
    def __init__(self, model, type_mapping, threshold=0.5, max_length=512):
        _validate_type_mapping(type_mapping)
        if type(max_length) is not int or max_length < 3:
            raise ValueError('max_length должен быть целым числом не меньше 3')
        # Общая проверка порога без вызова модели.
        predict_entities(model, '', type_mapping, threshold)
        self.model = model
        self.type_mapping = dict(type_mapping)
        self.threshold = threshold
        self.max_length = max_length

    def split_text(self, text):
        if not isinstance(text, str):
            raise ValueError('Текст должен быть строкой')
        offsets = [(start,end) for _,start,end in self.model.data_processor.words_splitter(text)]
        labels = list(self.type_mapping.values())
        def fits(start, end):
            words, max_words, tokens, limit = _input_dimensions(
                self.model, text[start:end], labels, self.max_length)
            return words <= max_words and tokens <= limit
        return chunk_spans(text, offsets, self.model.config.max_len, fits=fits)

    def predict_single(self, text, return_spans=True):
        entities = []
        for chunk in self.split_text(text):
            start, end = chunk['chunk_start'], chunk['chunk_end']
            for entity in predict_entities(self.model, text[start:end], self.type_mapping, self.threshold):
                entity = dict(entity)
                entity['start'] += start
                entity['end'] += start
                entities.append(entity)
        if not return_spans:
            return [{k:v for k,v in entity.items() if k not in ('start','end')} for entity in entities]
        return entities

    def predict_batch(self, texts, return_spans=True):
        return [self.predict_single(text, return_spans) for text in texts]
