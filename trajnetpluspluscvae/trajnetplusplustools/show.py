from contextlib import contextmanager


@contextmanager
def predicted_paths(*args, **kwargs):
    del args, kwargs
    yield None


@contextmanager
def canvas(*args, **kwargs):
    del args, kwargs
    yield None
