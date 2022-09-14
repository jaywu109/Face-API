import os
from pathlib import Path as _Path
from pathlib import PosixPath, WindowsPath


def resolve_path(paths):
    res = []
    for path in paths:
        for p in path.replace("\\", "/").split("/"):
            res.append(p)
    return res


class Path(type(_Path())):

    path_to_models = ""

    def __new__(cls, *args, **kwargs):
        args = resolve_path(args)
        args = (cls.path_to_models,) + tuple(args)
        if cls is (Path, "ModelPath"):
            cls = WindowsPath if os.name == "nt" else PosixPath
        self = cls._from_parts(args, init=False)
        if not self._flavour.is_supported:
            raise NotImplementedError("cannot instantiate %r on your system" % (cls.__name__,))
        self._init()
        return self
