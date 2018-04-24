import inspect
import os
import pickle
import logging


def cache():
    return _MemoryStorage()


def store(storage_directory=os.path.join(os.path.dirname(__file__), '..', 'output'), ignore=()):
    return _DiskStorage(storage_directory=storage_directory, ignore=ignore)


def get_function_identifier(function, call_args):
    function_identifier = os.path.join(function.__module__ + '.' + function.__name__,
                                       ','.join('{}={}'.format(key, value) for key, value in call_args.items()))
    return function_identifier


class _Storage(object):
    def __init__(self, ignore=()):
        self.ignore = ignore
        self._logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def __call__(self, function):
        def wrapper(*args, **kwargs):
            call_args = inspect.getcallargs(function, *args, **kwargs)
            call_args = {key: value for key, value in call_args.items() if key not in self.ignore}
            function_identifier = get_function_identifier(function, call_args)
            if self.is_stored(function_identifier):
                self._logger.debug("Loading from storage: {}".format(function_identifier))
                return self.load(function_identifier)
            result = function(*args, **kwargs)
            self._logger.debug("Saving to storage: {}".format(function_identifier))
            self.save(result, function_identifier)
            return result

        return wrapper

    def is_stored(self, function_identifier):
        raise NotImplementedError()

    def load(self, function_identifier):
        raise NotImplementedError()

    def save(self, result, function_identifier):
        raise NotImplementedError()


class _DiskStorage(_Storage):
    def __init__(self, storage_directory, ignore=()):
        super().__init__(ignore=ignore)
        self.storage_directory = storage_directory

    def storage_path(self, function_identifier):
        return os.path.join(self.storage_directory, function_identifier + '.pkl')

    def save(self, result, function_identifier):
        path = self.storage_path(function_identifier)
        path_dir = os.path.dirname(path)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir, exist_ok=True)
        savepath_part = path + '.filepart'
        with open(savepath_part, 'wb') as f:
            pickle.dump({'data': result}, f)
        os.rename(savepath_part, path)

    def is_stored(self, function_identifier):
        storage_path = self.storage_path(function_identifier)
        return os.path.isfile(storage_path)

    def load(self, function_identifier):
        path = self.storage_path(function_identifier)
        assert os.path.isfile(path)
        with open(path, 'rb') as f:
            return pickle.load(f)['data']


class _MemoryStorage(_Storage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = dict()

    def save(self, result, function_identifier):
        self.cache[function_identifier] = result

    def is_stored(self, function_identifier):
        return function_identifier in self.cache

    def load(self, function_identifier):
        return self.cache[function_identifier]
