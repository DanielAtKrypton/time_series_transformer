import torch
import pytest

def cuda_check(device):
    if device == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.empty_cache()    
        else:
            pytest.skip("needs a CUDA compatible GPU available to run this test")
        