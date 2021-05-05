try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"



from .dockwidgets import napari_experimental_provide_dock_widget
from .functionwidgets import napari_experimental_provide_function
