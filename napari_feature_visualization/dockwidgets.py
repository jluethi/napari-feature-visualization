from napari_plugin_engine import napari_hook_implementation
from .feature_vis import feature_vis
from .classifier_widgets import initialize_classifier, load_classifier

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    #return [ExampleQWidget, example_magic_widget, feature_vis]
    return [feature_vis, initialize_classifier, load_classifier]
