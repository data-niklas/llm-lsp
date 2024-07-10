from dataclasses import dataclass


@dataclass
class LspGenerationConfig:
    comments_processor: bool = True
    boundary_processor: bool = True
    lsp_processor: bool = True
    enabled: bool = (
        True  # quick setting to disable all processors, overrides other settings
    )

    def is_disabled(self) -> bool:
        return not self.enabled or not (
            self.comments_processor or self.boundary_processor or self.lsp_processor
        )
