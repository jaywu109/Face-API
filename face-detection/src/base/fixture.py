from random import shuffle

import diskcache as dc


disk_cache = dc.Cache("tmp")
memory_cache = {}
_first = False


def loadtest_fixture(name, extra_params=None, use_diskcache=False):
    def decorator(func):
        def wrap():
            cache = memory_cache if not use_diskcache else disk_cache
            if not _first:
                # Clear disk cache
                disk_cache.clear()

            if cache.get(name) is not None:
                # Get data in cache by args `name`
                data = cache.get(name)
            else:
                # Get data by `func` and save cache
                data = func()
                disk_cache[name] = data
            if extra_params is not None:
                data = sum([[dict(**i, **j) for i in data] for j in extra_params], [])
            shuffle(data)
            return data

        return wrap

    return decorator
