import inspect
import itertools
import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import xarray as xr

from mkgu.assemblies import merge_data_arrays


def get_function_identifier(function, call_args):
    function_identifier = os.path.join(function.__module__ + '.' + function.__name__,
                                       ','.join('{}={}'.format(key, value) for key, value in call_args.items()))
    return function_identifier


class _Storage(object):
    def __init__(self, filename_ignore=()):
        self.ignore = filename_ignore
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
    def __init__(self, storage_directory=os.path.join(os.path.dirname(__file__), '..', 'output'), filename_ignore=()):
        super().__init__(filename_ignore=filename_ignore)
        self.storage_directory = storage_directory

    def storage_path(self, function_identifier):
        return os.path.join(self.storage_directory, function_identifier + '.pkl')

    def save(self, result, function_identifier):
        path = self.storage_path(function_identifier)
        path_dir = os.path.dirname(path)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir, exist_ok=True)
        savepath_part = path + '.filepart'
        self.save_file(result, savepath_part)
        os.rename(savepath_part, path)

    def save_file(self, result, savepath_part):
        with open(savepath_part, 'wb') as f:
            pickle.dump({'data': result}, f)

    def is_stored(self, function_identifier):
        storage_path = self.storage_path(function_identifier)
        return os.path.isfile(storage_path)

    def load(self, function_identifier):
        path = self.storage_path(function_identifier)
        assert os.path.isfile(path)
        return self.load_file(path)

    def load_file(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)['data']


class _XarrayStorage(_DiskStorage):
    """
    All things in filename_ignore are combined into one file and loaded lazily
    """

    def __call__(self, function):
        def wrapper(*args, **kwargs):
            call_args = inspect.getcallargs(function, *args, **kwargs)
            identifier_callargs = {key: value for key, value in call_args.items() if key not in self.ignore}
            function_identifier = get_function_identifier(function, identifier_callargs)
            stored_result, reduced_call_args = None, call_args
            if self.is_stored(function_identifier):
                self._logger.debug("Loading from storage: {}".format(function_identifier))
                stored_result = self.load(function_identifier)
                reduced_call_args = self.filter_coords(call_args, stored_result)
                if len(reduced_call_args) == 0:
                    return stored_result
                else:
                    self._logger.debug("Computing missing: {}".format(reduced_call_args))
            result = function(**reduced_call_args)
            if stored_result:
                result = merge_data_arrays([stored_result, result])
            assert len(self.filter_coords(call_args, result)) == 0  # make sure coords are set equal to call_args
            result = self.filter_data(result, call_args)
            self._logger.debug("Saving to storage: {}".format(function_identifier))
            self.save(result, function_identifier)
            return result

        return wrapper

    def filter_coords(self, call_args, result):
        iter_call_args, non_iter_values = self._make_iterable(call_args)
        combinations = [dict(zip(iter_call_args, values)) for values in itertools.product(*iter_call_args.values())]
        uncomputed_combinations = []
        for combination in combinations:
            combination_result = result
            combination_result = self.filter_data(combination_result, combination)
            if len(combination_result) == 0:
                uncomputed_combinations.append(combination)
        if len(uncomputed_combinations) == 0:
            return {}
        return self._combine_call_args(uncomputed_combinations, non_iter_values)

    def filter_data(self, data, coords):
        for coord, coord_value in coords.items():
            if not hasattr(data, coord):
                raise ValueError("Coord not found in data: {}".format(coord))
            indexer = np.array([val == coord_value for val in data[coord].values])
            coord_dims = data[coord].dims
            dim_indexes = {dim: slice(None) if dim not in coord_dims else np.where(indexer)[0]
                           for dim in data.dims}
            data = data.isel(**dim_indexes)
        return data

    def _make_iterable(self, call_args):
        non_iter_values = []
        iter_call_args = {}
        for key, value in call_args.items():
            try:
                # Note: this won't package a single-value list into a list of lists
                iter(value)
            except TypeError:
                non_iter_values.append(key)
                value = [value]
            iter_call_args[key] = value
        return iter_call_args, non_iter_values

    def _combine_call_args(self, uncomputed_combinations, non_iter_values):
        call_args = defaultdict(list)
        for combination in uncomputed_combinations:
            for key, value in combination.items():
                call_args[key].append(value)
        for non_iter_value in non_iter_values:
            assert len(call_args[non_iter_value]) == 1
            call_args[non_iter_value] = call_args[non_iter_value][0]
        return call_args

    def storage_path(self, function_identifier):
        return os.path.join(self.storage_directory, function_identifier + '.nc')

    def save_file(self, result, savepath_part):
        result.to_netcdf(savepath_part)

    def load_file(self, path):
        return xr.open_dataarray(path)


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


cache = _MemoryStorage
store = _DiskStorage
store_xarray = _XarrayStorage
