from pathlib import PurePath

from ._utils import Path, _Path


class ModelPath(Path):
    """Inherits of pathlib.Path, load models data in `src/models/`.

    Args:
        path (str): Enter the path in order.

    Example:
        >>> models_dirt = ModelPath() # Root of ModelPath
        >>> print("models/: ", models_dirt)
        models/:  /Users/shemyu/Documents/Wisers/py-api-project/face-detection-2022/src/models
        >>> data_txt = models_dirt / "data.txt"
        >>> print("path of data.txt: ", data_txt)
        path of data.txt:  /Users/shemyu/Documents/Wisers/py-api-project/face-detection-2022/src/models/data.txt
        >>> print("data.txt is file? ", data_txt.is_file())
        data.txt is file?  False
        >>> tiny_archi_sample = ModelPath("tiny_model.sample", "archi.sample")
        >>> print("path of archi.sample: ", tiny_archi_sample)
        path of archi.sample:  /Users/shemyu/Documents/Wisers/py-api-project/face-detection-2022/src/models/tiny_model.sample/archi.sample
        >>> print("Is archi.sample exists: ", tiny_archi_sample.exists())
        Is archi.sample exists:  True
    """

    path_to_models = _Path(PurePath(__file__).parent.parent, "models").resolve()
