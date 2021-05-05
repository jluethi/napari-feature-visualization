# Util functions for plugins
from functools import lru_cache
import pandas as pd
from enum import Enum

@lru_cache(maxsize=16)
def get_df(path):
    return pd.read_csv(path)


class ColormapChoices(Enum):
    viridis='viridis'
    plasma='plasma'
    inferno='inferno'
    magma='magma'
    cividis='cividis'
    Greys='Greys'
    Purples='Purples'
    Blues='Blues'
    Greens='Greens'
    Oranges='Oranges'
    Reds='Reds'
    YlOrBr='YlOrBr'
    YlOrRd='YlOrRd'
    OrRd='OrRd'
    PuRd='PuRd'
    RdPu='RdPu'
    BuPu='BuPu'
    GnBu='GnBu'
    PuBu='PuBu'
    YlGnBu='YlGnBu'
    PuBuGn='PuBuGn'
    BuGn='BuGn'
    YlGn='YlGn'
    PiYG='PiYG'
    PRGn='PRGn'
    BrBG='BrBG'
    PuOr='PuOr'
    RdGy='RdGy'
    RdBu='RdBu'
    RdYlBu='RdYlBu'
    RdYlGn='RdYlGn'
    Spectral='Spectral'
    coolwarm='coolwarm'
    bwr='bwr'
    seismic='seismic'
    turbo='turbo'
    jet='jet'
