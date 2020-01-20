import os
import errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5 (except OSError, exc: for Python <2.5)
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


if __name__=='__main__':
    mkdir_p('/tmp/test')
    mkdir_p('/tmp/holmes')