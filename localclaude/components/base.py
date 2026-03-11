import abc
import optuna
from typing import Dict, Any, Type
from transformers import PreTrainedModel, LlamaConfig

MUTATOR_REGISTRY: Dict[str, Type['BaseMutator']] = {}

def register_mutator(name: str):
    def wrapper(cls: Type['BaseMutator']):
        MUTATOR_REGISTRY[name] = cls
        return cls
    return wrapper

class BaseMutator(abc.ABC):    
    @abc.abstractmethod
    def build_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        定义当前组件向 Optuna 注册的搜索空间。
        必须返回一个字典，键为超参名，值为 trial.suggest_xxx() 的结果。
        """
        pass

    @abc.abstractmethod
    def mutate(self, model: PreTrainedModel, config: LlamaConfig, params: Dict[str, Any]) -> PreTrainedModel:
        """
        执行架构的动态突变，必须保证 weight preserving。
        """
        pass