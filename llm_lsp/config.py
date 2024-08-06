from dataclasses import dataclass
from typing import Optional

@dataclass
class LspGenerationConfig:
    comments_processor: bool = True
    boundary_processor: bool = True
    lsp_processor: bool = True
    chat_history_log_file: Optional[str] = None
    predict_correct_completion_symbol: bool = True
    force_custom_pad: bool = False
    masked_gen: bool = True
    use_completion_context: bool = True
    use_deprecation_context: bool = True
    use_signature_context: bool = True
    enabled: bool = (
        True  # quick setting to disable all processors, overrides other settings
    )

    def is_disabled(self) -> bool:
        return not self.enabled or not (
            self.comments_processor or self.boundary_processor or self.lsp_processor
        )
