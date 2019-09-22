# -*- coding: utf-8 -*-
import inspect
import os
import argparse
import logging


LOGGING_LEVELS = [
    ('debug', logging.DEBUG),
    ('info', logging.INFO),
    ('warning', logging.WARNING),
    ('error', logging.ERROR),
    ('critical', logging.CRITICAL)
]


def log_level_arg(string):
    s = string.lower()
    for k, lvl in LOGGING_LEVELS:
        if k.startswith(s):
            return lvl
    raise argparse.ArgumentTypeError('Not a valid log level %s' % string)


# for matplotlib
def valid_plot_ext_arg(string):  # TODO: could use some work
    if string.startswith('.'):
        string = string[1:]
    return string.lower()


# Similar to argparse.FileType in function, but does not open/create anything

def valid_existing_dir_arg(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError('Directory does not exist: %s' % path)
    elif not os.path.isdir(path):
        raise argparse.ArgumentTypeError('Not a directory: %s' % path)
    elif not os.access(path, os.R_OK):
        raise argparse.ArgumentTypeError('Directory is not readable: %s' % path)
    return path


def valid_new_dir_arg(path):
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise argparse.ArgumentTypeError('Not a directory: %s' % path)
        if not os.access(path, os.W_OK | os.X_OK):
            raise argparse.ArgumentTypeError('Path is not writable: %s' % path)
    else:
        base, last = os.path.split(path)
        if len(base) == 0:
            base = '.'
        if not os.path.exists(base):
            raise argparse.ArgumentTypeError('Containing directory does not exist: %s' % base)
        if not os.access(base, os.W_OK | os.X_OK):
            raise argparse.ArgumentTypeError('Path is not writable: %s' % base)
    return path


def valid_existing_file_arg(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError('File does not exist: %s' % path)
    elif not os.path.isfile(path):
        raise argparse.ArgumentTypeError('Not a regular file: %s' % path)
    elif not os.access(path, os.R_OK):
        raise argparse.ArgumentTypeError('File is not readable: %s' % path)
    return path


def valid_new_file_arg(path):    
    if os.path.exists(path):
        if not os.path.isfile(path):
            raise argparse.ArgumentTypeError('Not a regular file: %s' % path)
    else:
        base, last = os.path.split(path)
        if len(base) == 0:
            base = '.'
        if not os.path.exists(base):
            raise argparse.ArgumentTypeError('Directory does not exist: %s' % base)
    if not os.access(path, os.W_OK | os.X_OK):
        raise argparse.ArgumentTypeError('Path is not writable: %s' % path)
    return path


def logger_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-l', '--loglevel',
                        type=log_level_arg,
                        help='Python logging level (case-insensitive, first letter suffuces).')
    parser.add_argument('--logfile',
                        type=valid_new_file_arg,
                        help='Log file name.')
    return parser


def setup_logger(args=None, loglevel=None, logfile=None):
    if args is not None:
        loglevel = args.loglevel
        logfile = args.logfile
    logging.basicConfig(
        level=loglevel,
        filename=logfile
    )


def pass_args_to_fun(fun, args, **xtra_args):
    args_dict = vars(args)
    args_dict.update(xtra_args)
    sig = inspect.signature(fun)
    args_dict = {k: v for k, v in args_dict.items() if k in sig.parameters}
    ba = sig.bind(**args_dict)
    return fun(*ba.args, **ba.kwargs)