import copy
import unittest

from src.config import load_config, validate_config, PROJECT_ROOT


class ConfigTests(unittest.TestCase):
    def test_default_schema_has_29_types_and_59_bio_labels(self):
        config = load_config()
        self.assertEqual(len(config['data']['entity_types']), 29)
        self.assertEqual(config['model']['num_labels'], 59)
        self.assertIn('STATE_OR_PROVINCE', config['data']['entity_types'])
        self.assertNotIn('STATE_OR_PROV', config['data']['entity_types'])

    def test_validation_does_not_mutate_input_and_resolves_paths(self):
        config = load_config()
        config['paths']['models'] = 'custom-models'
        original = copy.deepcopy(config)
        resolved = validate_config(config)
        self.assertEqual(config, original)
        self.assertEqual(resolved['paths']['models'], str(PROJECT_ROOT / 'custom-models'))


