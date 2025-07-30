from .. import BACKEND

if BACKEND == 'torchsparse':
    from .conv_torchsparse import *
elif BACKEND == 'spconv':
    from .conv_spconv import *