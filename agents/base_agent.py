from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name: str, cost: float):
        self.name = name
        self.cost = cost

    @abstractmethod
    def act(self, query: str) -> dict:
        pass
