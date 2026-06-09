"""FOCUS dataset class for mmrotate.
"""

import os.path as osp

try:
    from mmrotate.datasets import DOTADataset  # type: ignore
    from mmrotate.registry import DATASETS  # type: ignore
    _HAS_MMROTATE = True
except Exception:  # pragma: no cover
    DOTADataset = object
    DATASETS = None
    _HAS_MMROTATE = False


# ----------------------------------------------------------------------------

if _HAS_MMROTATE:

    @DATASETS.register_module()
    class FOCUSDataset(DOTADataset):
        """FOCUS four-chamber-view fetal cardiac detection dataset.

        Class order MATTERS - it is the ID order the detector head and the
        AP evaluator use. We put ``thorax`` first (id=0) and ``cardiac``
        second (id=1) to match alphabetical convention and to make the CTR
        metric's class lookup unambiguous.

        Wiring required by the parent ``DOTADataset.load_data_list``:
            - ``ann_file`` MUST be a directory of ``.txt`` annotation files
              (not the empty string - that path emits empty instances and
              the dataset becomes inference-only).
            - ``data_prefix['img_path']`` MUST be the directory of ``.png``
              images; the image basename is derived from the txt stem
              (``train_001.txt`` -> ``train_001.png``).

        ``DOTADataset.__init__`` signature (mmrotate 1.0.0rc1):
            ``(img_shape=(1024, 1024), diff_thr=100, **kwargs)``
        No ``img_suffix`` kwarg - the parent hardcodes ``.png`` in
        ``load_data_list``. Our prepared FOCUS images are all PNG so this
        is fine.
        """

        METAINFO = {
            "classes":  ("thorax", "cardiac"),
            "palette":  [(220, 20, 60), (0, 128, 255)],
        }

else:  # pragma: no cover

    class FOCUSDataset:  # type: ignore[no-redef]
        """Placeholder when mmrotate is not installed.

        Importing this module without mmrotate must not crash, because tests
        and the CTR pipeline can run standalone. Instantiating ``FOCUSDataset``
        in that environment raises explicitly.
        """

        METAINFO = {
            "classes": ("thorax", "cardiac"),
            "palette": [(220, 20, 60), (0, 128, 255)],
        }

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "FOCUSDataset requires mmrotate. Install via "
                "`mim install \"mmrotate>=1.0.0rc1\"` inside conda env "
                "`openus-mmrot`."
            )
