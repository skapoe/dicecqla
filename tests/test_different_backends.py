from main import argparse_default
from core.executer import Execute
import sys
import pytest


class TestPolyak:
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pandas_as_backend(self):
        args = argparse_default([])
        args.path_dataset_folder = 'KGs/UMLS'
        args.backend = 'pandas'
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()


    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pandas_as_backend(self):
        args = argparse_default([])
        args.path_dataset_folder = 'KGs/UMLS'
        args.backend = 'modin'
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_pandas_as_backend(self):
        args = argparse_default([])
        args.path_dataset_folder = 'KGs/UMLS'
        args.backend = 'polars'
        args.trainer = 'torchCPUTrainer'
        Execute(args).start()
