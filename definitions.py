from typing import NamedTuple
from enum import Enum, EnumMeta, auto


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
    EQ = 'equal'
    MKTCAP = 'mktcap'
    RISKPAR = 'riskpar'
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

    # IMV = 'imv'


class Strategy(NamedTuple):
    dataset: StockUniverses
    algo: Algorithms
    wght_mtd: WeightMethods
    n_stk: int
    n_obs: int


class TopStrategies(Enum):
    SP500 = Strategy(dataset=StockUniverses.SP500,
                     algo=Algorithms.SEV,
                     wght_mtd=WeightMethods.RISKPAR,
                     n_stk=10,
                     n_obs=90)
    STOXXE600 = Strategy(dataset=StockUniverses.STOXXE600,
                         algo=Algorithms.RMLIN,
                         wght_mtd=WeightMethods.RISKPAR,
                         n_stk=10,
                         n_obs=60)


class CorrelationMethods(Enum):
    LINEAR = auto()
    SPEARMAN = auto()
    SEV = auto()
