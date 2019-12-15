import sys

__version__ = '0.1.5'


# python version
_p3 = False
if sys.version_info > (3, 0):
    _p3 = True
del(sys)

__all__ = [
    'utils',
    'fourier_tools',
    'processing',
    'io',
    'quantify',
    'bayes']

for x in __all__:
    exec('from . import %s' %(x))
del(x)