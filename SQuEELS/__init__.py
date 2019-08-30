import sys

name = "SQuEELS"
__version__ = '0.1.1'


# python version
_p3 = False
if sys.version_info > (3, 0):
    _p3 = True
del(sys)

__all__ = [
    'fourier_tools',
    'quantify',
    'processing']

for x in __all__:
    exec('from . import %s' %(x))
del(x)