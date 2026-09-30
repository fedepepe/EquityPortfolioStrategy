from typing import NamedTuple, List, Union, Optional
from enum import Enum, EnumMeta, auto
import numpy as np

class EnumDirectValueMeta(EnumMeta):
    def __getattribute__(cls, name):
        value = super().__getattribute__(name)
        if isinstance(value, cls):
            value = value.value
        return value


class StockUniverses(Enum, metaclass=EnumDirectValueMeta):
    SP500 = 'SP500'
    STOXXE600 = 'STOXXE600'


class WeightMethods(Enum, metaclass=EnumDirectValueMeta):
    METRIC = 'metric'
    EQ = 'equal'
    MKTCAP = 'mktcap'
    RISK_PARITY = 'riskpar'
    LOTP = 'lotp'
    ILOTP = 'ilotp'


class Algorithms(Enum, metaclass=EnumDirectValueMeta):
    SEV = 'sev'
    RMSEV = 'rmsev'

    LIN = 'lin'
    RMLIN = 'rmlin'
    LOLIN = 'lolin'
    LORMLIN = 'lormlin'

    MTM = 'mtm'
    RMMTM = 'rmmtm'
    LOMTM = 'lomtm'
    LORMMTM = 'lormmtm'

    MAX_SHARPE = 'max_sr'

    # IMV = 'imv'


class Strategy(NamedTuple):
    dataset: StockUniverses
    algo: Algorithms
    wght_mtds: Union[WeightMethods, List[WeightMethods]]
    n_stk: Optional[int]
    n_obs: Optional[Union[int, np.array]] = None


class TopStrategies(Enum):
    SP500 = Strategy(dataset=StockUniverses.SP500,
                     algo=Algorithms.SEV,
                     wght_mtds=WeightMethods.RISK_PARITY,
                     n_stk=10,
                     n_obs=90)
    STOXXE600 = Strategy(dataset=StockUniverses.STOXXE600,
                         algo=Algorithms.RMLIN,
                         wght_mtds=WeightMethods.RISK_PARITY,
                         n_stk=10,
                         n_obs=60)


class CorrelationMethods(Enum):
    LINEAR = auto()
    SPEARMAN = auto()
    SEV = auto()
