try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"



from .feature_vis import napari_experimental_provide_dock_widget
from ._regionprops import napari_experimental_provide_function