from enum import Enum, EnumMeta


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
    MTM = 'mtm'
    RMMTM = 'rmmtm'
    LOMTM = 'lomtm'
    LORMMTM = 'lormmtm'
    LIN = 'lin'
    RMLIN = 'rmlin'
    LOLIN = 'lolin'
    LORMLIN = 'lormlin'
    SEV = 'sev'
    RMSEV = 'rmsev'
