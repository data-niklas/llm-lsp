from contextlib import contextmanager

import torch

# from llm_lsp.lsp_client import LspClient



class PipelineMixin:
    """Requires a device argument"""

    @contextmanager
    def device_placement(self):
        """
        Context Manager allowing tensor allocation on the user-specified device in framework agnostic way.

        Returns:
            Context manager

        Examples:

        ```python
        # Explicitly ask for tensor allocation on CUDA device :0
        pipe = pipeline(..., device=0)
        with pipe.device_placement():
            # Every framework specific tensor allocation will be done on the request device
            output = pipe(...)
        ```"""
        with torch.device(self.device):
            yield
