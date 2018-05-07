import os
import tempfile

import numpy as np
import pytest
import xarray as xr
from neurality.storage import store_xarray, store, cache


class TestXarrayStore:
    def test_same(self):
        function_called = False

        @store_xarray(storage_directory=tempfile.mkdtemp(), identifier_ignore=['x'], combine_fields=['x'])
        def func(x, base=1):
            nonlocal function_called
            assert not function_called
            function_called = True
            return xr.DataArray(x, coords={'x': x, 'base': ('x', [base])}, dims='x')

        def test():
            result = func([1])
            assert isinstance(result, xr.DataArray)
            assert result == 1

        test()
        # second call returns same thing and doesn't actually call function again
        test()

    def test_complimentary(self):
        @store_xarray(storage_directory=tempfile.mkdtemp(), identifier_ignore=['x'], combine_fields=['x'])
        def func(x, base=1):
            return xr.DataArray(x, coords={'x': x, 'base': ('x', [base])}, dims='x')

        np.testing.assert_array_equal(func([1]), 1)
        np.testing.assert_array_equal(func([2]), 2)

    def test_missing_coord(self):
        @store_xarray(storage_directory=tempfile.mkdtemp(), identifier_ignore=['x'], combine_fields=['x'])
        def func(x, base=1):
            return xr.DataArray(x, coords={'x': x}, dims='x')

        with pytest.raises(ValueError):
            func([1])

    def test_combined(self):
        expected_x = None

        @store_xarray(storage_directory=tempfile.mkdtemp(), identifier_ignore=['x'], combine_fields=['x'])
        def func(x, base=1):
            assert len(x) == 1 and x[0] == expected_x
            return xr.DataArray(x, coords={'x': x, 'base': ('x', [base])}, dims='x')

        expected_x = 1
        np.testing.assert_array_equal(func([1]), 1)
        expected_x = 2
        combined = func([1, 2])
        np.testing.assert_array_equal(combined, [1, 2])


class TestStore:
    def test(self):
        storage_dir = tempfile.mkdtemp()
        function_called = False

        @store(storage_directory=storage_dir)
        def func(x):
            nonlocal function_called
            assert not function_called
            function_called = True
            return x

        assert func(1) == 1
        assert os.path.isfile(os.path.join(storage_dir, 'test_storage.func', 'x=1.pkl'))
        # second call returns same thing and doesn't actually call function again
        assert func(1) == 1


class TestCache:
    def test(self):
        function_called = False

        @cache()
        def func(x):
            nonlocal function_called
            assert not function_called
            function_called = True
            return x

        assert func(1) == 1
        # second call returns same thing and doesn't actually call function again
        assert func(1) == 1
