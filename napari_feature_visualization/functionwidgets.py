from napari_plugin_engine import napari_hook_implementation
from ._regionprops import regionprops
#from .classifier import classifier_widget


@napari_hook_implementation
def napari_experimental_provide_function():
    return [regionprops]
    #return [regionprops, classifier_widget]
