from typing import List, Dict, Any
from abc import ABC, abstractmethod

class MessageFormatter(ABC):
    @abstractmethod
    def create_completion_messages(self, initial_code: str, system_prompt_enabled: bool, instruction_text: str) -> List[Dict[str, Any]]:
        pass