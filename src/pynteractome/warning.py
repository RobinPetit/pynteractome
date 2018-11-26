# std
from sys import stderr
# local
from .utils import log

_warning_mode = 'log'

def set_warning_mode(mode='log'):
    '''
    Set the behaviour of warnings. Available parameters:

    * ``'log'``: log the warnings on ``sys.stdout``
    * ``'stderr'``: log the warnings on ``sys.stderr``
    * ``'raise'``: raise a :class:`WarningException`
    * ``'ignore'``: to not do anything at all

    Args:
        mode (str):
            mode of the warnings: determine what behaviour is chosen when a warning is set
    '''
    global _warning_mode
    available_modes = ('log', 'stderr', 'raise', 'ignore')
    if mode not in available_modes:
        raise ValueError('Warning mode not recognized: ' + mode)
    _warning_mode = mode

def warning(path):
    message = '[Warning]:\t' + path
    if _warning_mode == 'log':
        log(message)
    elif _warning_mode == 'stderr':
        log(message, file_=stderr)
    elif _warning_mode == 'raise':
        raise WarningException(message)

class WarningException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)  # Maybe complete or check parent class
