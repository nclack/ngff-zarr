"""This module is a napari plugin.

It implements the ``napari_get_reader`` hook specification, (to create a reader plugin).
"""

import logging
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union, Tuple, Callable

import numpy as np
from ngff_zarr import from_ngff_zarr
from ngff_zarr.ngff_image import NgffImage
from ngff_zarr.to_multiscales import Multiscales

# Type aliases to avoid requiring napari as a dependency
LayerData = Tuple[Any, Dict[str, Any], str]  # (data, metadata, layer_type)
ReaderFunction = Callable[..., List[LayerData]]

METADATA_KEYS = ("name", "visible", "contrast_limits", "colormap", "metadata")

# major and minor versions as int
napari_version = tuple(map(int, list(version("napari").split(".")[:2])))

LOGGER = logging.getLogger("napari_ngff_zarr.reader")


def napari_get_reader(path: Union[str, List[str]]) -> Optional[ReaderFunction]:
    """Returns a reader for supported paths that include IDR ID.

    - URL of the form: https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/ID.zarr/
    """
    if isinstance(path, list):
        if len(path) > 1:
            warnings.warn("more than one path is not currently supported")
        path = path[0]
    
    try:
        multiscales = from_ngff_zarr(path)
        return transform(multiscales)
    except Exception as e:
        LOGGER.debug(f"Failed to read {path} as ngff_zarr: {e}")
        return None


def transform(multiscales: Multiscales) -> ReaderFunction:
    def f(*args: Any, **kwargs: Any) -> List[LayerData]:
        results: List[LayerData] = list()

        for image in multiscales.images:
            data = image.data
            metadata: Dict[str, Any] = {}
            
            if data is None:
                LOGGER.debug("skipping non-data %s", image)
                continue
                
            layer_type: str = "image"
            channel_axis = None
            
            # Find channel axis if it exists
            for i, axis in enumerate(multiscales.metadata.axes):
                if axis.type == "channel":
                    channel_axis = i
                    break

            # Add scale and translation from coordinate transformations
            if image.scale:
                metadata["scale"] = tuple(image.scale.values())
            if image.translation:
                metadata["translate"] = tuple(image.translation.values())

            # Handle labels vs image
            if multiscales.metadata.omero is None:
                layer_type = "labels"
                for x in METADATA_KEYS:
                    if hasattr(multiscales.metadata, x):
                        metadata[x] = getattr(multiscales.metadata, x)
            else:
                if channel_axis is not None:
                    # multi-channel; Copy known metadata values
                    metadata["channel_axis"] = channel_axis
                    for x in METADATA_KEYS:
                        if hasattr(multiscales.metadata, x):
                            metadata[x] = getattr(multiscales.metadata, x)
                    # Handle channel names
                    if multiscales.metadata.omero and multiscales.metadata.omero.channels:
                        metadata["name"] = [ch.label for ch in multiscales.metadata.omero.channels]
                        metadata["contrast_limits"] = [
                            (ch.window.min, ch.window.max) for ch in multiscales.metadata.omero.channels
                        ]
                        metadata["colormap"] = [ch.color for ch in multiscales.metadata.omero.channels]
                else:
                    # single channel image
                    for x in METADATA_KEYS:
                        if hasattr(multiscales.metadata, x):
                            try:
                                metadata[x] = getattr(multiscales.metadata, x)
                            except Exception:
                                pass

            rv: LayerData = (data, metadata, layer_type)
            results.append(rv)

        return results

    return f
