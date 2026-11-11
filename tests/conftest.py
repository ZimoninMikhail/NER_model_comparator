"""Одна команда pytest tests; ни сбор, ни проверки не обращаются к Hub."""
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('MPLBACKEND','Agg')
