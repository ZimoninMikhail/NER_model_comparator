import unittest
from unittest.mock import patch

from src.device import resolve_device


class DeviceTests(unittest.TestCase):

    def test_auto_uses_cuda_when_available(self):
        with patch('torch.cuda.is_available',return_value=True):
            self.assertEqual(resolve_device('auto'),'cuda')
            self.assertEqual(resolve_device('cuda'),'cuda')

    def test_explicit_cpu_does_not_probe_gpu(self):
        with patch('torch.cuda.is_available') as available:
            self.assertEqual(resolve_device('cpu'),'cpu')
            available.assert_not_called()

    def test_explicit_unavailable_cuda_is_an_error(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(ValueError,'GPU недоступен'):resolve_device('cuda')


