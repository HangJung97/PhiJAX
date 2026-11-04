from typing import TYPE_CHECKING

from phijax._lazy import lazy_exports

if TYPE_CHECKING:
    from phijax.data.artifacts import (
        PredictionArtifact,
        load_prediction_artifact,
        save_prediction_artifact,
        to_matlab_prediction_arrays,
    )
    from phijax.data.batching import BatchSize, DevicePool
    from phijax.data.builders import build_array_pools
    from phijax.data.datamodule import DataStage, PhiDataModule
    from phijax.data.io import ArrayFormat, get_array, load_arrays
    from phijax.data.pools import HostPool, input_statistics, reconstruct_predictions
    from phijax.data.samplers import (
        BatchSampler,
        RandomRowSampler,
        SpaceTimeSampler,
        UniformDomainSampler,
        create_sampler,
    )
    from phijax.data.sources import (
        ChunkedPredictionSource,
        NamedBatchSource,
        PredictionBatchSource,
        TrainingBatchSource,
    )
    from phijax.data.transforms import log_compress, minmax_scale, scale_by_max, standardize

__getattr__, __dir__ = lazy_exports(
    __name__,
    {
        "ArrayFormat": "phijax.data.io",
        "BatchSampler": "phijax.data.samplers",
        "BatchSize": "phijax.data.batching",
        "ChunkedPredictionSource": "phijax.data.sources",
        "DataStage": "phijax.data.datamodule",
        "DevicePool": "phijax.data.batching",
        "HostPool": "phijax.data.pools",
        "NamedBatchSource": "phijax.data.sources",
        "PhiDataModule": "phijax.data.datamodule",
        "PredictionArtifact": "phijax.data.artifacts",
        "PredictionBatchSource": "phijax.data.sources",
        "RandomRowSampler": "phijax.data.samplers",
        "SpaceTimeSampler": "phijax.data.samplers",
        "TrainingBatchSource": "phijax.data.sources",
        "UniformDomainSampler": "phijax.data.samplers",
        "build_array_pools": "phijax.data.builders",
        "create_sampler": "phijax.data.samplers",
        "get_array": "phijax.data.io",
        "input_statistics": "phijax.data.pools",
        "load_arrays": "phijax.data.io",
        "load_prediction_artifact": "phijax.data.artifacts",
        "log_compress": "phijax.data.transforms",
        "minmax_scale": "phijax.data.transforms",
        "reconstruct_predictions": "phijax.data.pools",
        "save_prediction_artifact": "phijax.data.artifacts",
        "scale_by_max": "phijax.data.transforms",
        "standardize": "phijax.data.transforms",
        "to_matlab_prediction_arrays": "phijax.data.artifacts",
    },
)

__all__ = [
    "ArrayFormat",
    "BatchSampler",
    "BatchSize",
    "ChunkedPredictionSource",
    "DataStage",
    "DevicePool",
    "HostPool",
    "NamedBatchSource",
    "PhiDataModule",
    "PredictionArtifact",
    "PredictionBatchSource",
    "RandomRowSampler",
    "SpaceTimeSampler",
    "TrainingBatchSource",
    "UniformDomainSampler",
    "build_array_pools",
    "create_sampler",
    "get_array",
    "input_statistics",
    "load_arrays",
    "load_prediction_artifact",
    "log_compress",
    "minmax_scale",
    "reconstruct_predictions",
    "save_prediction_artifact",
    "scale_by_max",
    "standardize",
    "to_matlab_prediction_arrays",
]
