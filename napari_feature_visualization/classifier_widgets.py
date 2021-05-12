from magicgui import magic_factory
from napari import Viewer
from magicgui import widgets
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.colors import ListedColormap
from .utils import get_df
from .classifier import Classifier

def _init_classifier(widget):
    def get_feature_choices(*args):
        try:
            df = get_df(widget.DataFrame.value)
            return list(df.columns)
        except IOError:
            return [""]

    # set feature and label_column "default choices"
    # to be a function that gets the column names of the
    # currently loaded dataframe
    widget.feature_selection._default_choices = get_feature_choices
    widget.label_column._default_choices = get_feature_choices

    @widget.DataFrame.changed.connect
    def update_df_columns(event):
        # event value will be the new path
        # get_df will give you the cached df
        # ...reset_choices() calls the "get_feature_choices" function above
        # to keep them updated with the current dataframe
        widget.feature_selection.reset_choices()
        widget.label_column.reset_choices()
        features = widget.label_column.choices
        if 'label' in features:
            widget.label_column.value = 'label'
        elif 'Label' in features:
            widget.label_column.value = 'Label'
        elif 'index' in features:
            widget.label_column.value = 'index'

        if 'additional_features' in widget.label_layer.value.properties:
            widget.additional_features.value = widget.label_layer.value.properties['additional_features']
        # TODO: Allow all features / also the initial feature to be set. Currently is being called before the lis

        if 'feature_selection' in widget.label_layer.value.properties:
            if widget.label_layer.value.properties['feature_selection'] in widget.feature_selection.choices:
                widget.feature_selection.value = widget.label_layer.value.properties['feature_selection']

    @widget.label_layer.changed.connect
    def update_paths(event):
        if 'DataFrame' in widget.label_layer.value.properties:
            widget.DataFrame.value = widget.label_layer.value.properties['DataFrame']


@magic_factory(
        call_button="Initialize Classifier",
        feature_selection = {"choices": [""]},
        label_column = {"choices": [""]},
        widget_init=_init_classifier,
        )
def initialize_classifier(viewer: Viewer,
                      label_layer: "napari.layers.Labels",
                      DataFrame: Path,
                      classifier_name = 'test',
                      feature_selection='',
                      additional_features='',
                      label_column=''):
    # TODO: Make feature selection a widget that allows multiple features to be selected, not just one
    # Something like this in QListWidget: https://stackoverflow.com/questions/4008649/qlistwidget-and-multiple-selection
    # See issue here: https://github.com/napari/magicgui/issues/229
    # TODO: Check that input features are numeric => do in classifier, deal with exception here
    training_features = [feature_selection]

    # Workaround: provide a text box to enter additional features separated by comma, parse them as well
    if additional_features:
        training_features += [x.strip() for x in additional_features.split(',')]

    site_df = get_df(DataFrame)
    site_df['path']=DataFrame
    index_columns=('path', label_column)
    site_df = site_df.set_index(list(index_columns))
    clf = Classifier(name=classifier_name, features=site_df, training_features=training_features, index_columns=index_columns)

    ClassifierWidget(clf, label_layer, DataFrame, viewer)


def _init_load_classifier(widget):
    # Trying to be smart with handling inputs
    # TODO: Add an option to check the current working directory for .clf files?
    #       As an option if no classifier_path is provided as a property
    # TODO: Check if a label layer is available, only do this then
    # Inputs always update with properties when label layer is changed.
    @widget.label_layer.changed.connect
    def update_paths(event):
        if 'classifier_path' in widget.label_layer.value.properties:
            widget.classifier_path.value = widget.label_layer.value.properties['classifier_path']
        if 'DataFrame' in widget.label_layer.value.properties:
            widget.DataFrame.value = widget.label_layer.value.properties['DataFrame']


@magic_factory(
        call_button="Load Classifier",
        widget_init=_init_load_classifier
        )
def load_classifier(viewer: Viewer,
                    label_layer: "napari.layers.Labels",
                    classifier_path: Path,
                    DataFrame: Path):
    # TODO: Refactor: Could this be integrated with the initialize widget, with optional inputs? Could be tricky
    # TODO: Make classifier path optional? If no path is added (empty path object?, None?), don't add data to the classifier, take the existing data?
    #       BUT: Needed for site identification atm => not possible to leave out
    # TODO: Add option to add new features to the classifier that were not added at initialization => unsure where to do this. Should it also be possible when initializing a classifier?
    # TODO: Add ability to see currently selected features
    # TODO: Ensure classifier path ends in .clf and DataFrame path ends in .csv
    # TODO: Can I take a guess for the dataframe path & name based on the filename of the site? Can I get the filename of the current site somehow? => probably not?
    # TODO: Idea for improved input handling: Out detect any .clf files in the current working directory. If any exist, pick the most recent one by default
    # TODO: If the label layer has properties for classifier_path and DataFrame, pick those => ways to parse inputs
    #classifier_name = classifier_path.stem

    with open(classifier_path, 'rb') as f:
        clf = pickle.loads(f.read())

    training_features = clf.training_features
    site_df = get_df(DataFrame)
    site_df['path']=DataFrame
    index_columns=clf.index_columns
    # Catches if new data frame doesn't contain the index columns
    assert all([index_column in site_df.columns for index_column in index_columns]), 'These two columns are not available in the current dataframe: {}'.format(index_columns)
    # TODO: Notify the user why the classifier is not loaded
    site_df = site_df.set_index(list(index_columns))

    clf.add_data(site_df, training_features=training_features, index_columns=index_columns)

    ClassifierWidget(clf, label_layer, DataFrame, viewer)


class ClassifierWidget:
    def __init__(self, clf, label_layer, DataFrame, viewer):
        self.clf = clf
        # TODO: Add a warning if a classifier with this name already exists => shall it be overwritten? => Confirmation box
        self.clf.save()
        self.label_layer = label_layer
        self.DataFrame = DataFrame
        self.viewer = viewer

        # Parameters for the colormaps
        # TODO: Generalize number of classes & colormap?
        self.nb_classes = 4
        self.cmap = ListedColormap([(0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0)])

        # Create a selection & prediction layer
        # TODO: Handle state when those layers were already created. Replace them otherwise?
        # https://napari.org/guides/stable/magicgui.html#updating-an-existing-layer
        self.prediction_layer = viewer.add_labels(label_layer.data, name='prediction', opacity=1.0, scale=label_layer.scale)
        self.selection_layer = viewer.add_labels(label_layer.data, name='selection', opacity=1.0, scale=label_layer.scale)
        self.colordict = self.create_label_colormap(self.selection_layer, clf.train_data, 'train')
        self.create_label_colormap(self.prediction_layer, clf.predict_data, 'predict')
        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(label_layer)

        widget = self.create_selector_widget(self.label_layer)

        # add widget to napari
        viewer.window.add_dock_widget(widget, area='right', name=clf.name)

    def create_selector_widget(self, label_layer):
        # TODO: Define a minimum number of selections. Below that, show a warning (training-test split can be very weird otherwise, e.g all 1 class)
        # TODO: Generalize this. Instead of 0, 1, 2: Arbitrary class numbers. Ability to add classes & name them?
        choices = ['Deselect', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
        # TODO: Make default choice Class 1
        selector = widgets.RadioButtons(choices=choices, label='Selection Class:')
        save_button = widgets.PushButton(value=True, text='Save Classifier')
        run_button = widgets.PushButton(value=True, text='Run Classifier')
        container = widgets.Container(widgets=[selector, save_button, run_button])
        # TODO: Add text field & button to save classifier output to disk

        @label_layer.mouse_drag_callbacks.append
        def toggle_label(obj, event):
            self.selection_layer.visible=True
            # Need to scale position that event.position returns by the label_layer scale.
            # If scale is (1, 1, 1), nothing changes
            # If scale is anything else, this makes the click still match the correct label
            scaled_position = tuple(pos / scale for pos, scale in zip(event.position, label_layer.scale))
            label = label_layer.get_value(scaled_position)
            #label = label_layer.get_value(event.position)
            # Check if background or foreground was clicked. If background was clicked, do nothing (background can't be assigned a class)
            if label == 0:
                pass
            else:
                # Check if the label exists in the current dataframe. Otherwise, do nothing
                if (self.DataFrame, label) in self.clf.train_data.index:
                    # Assign name of class
                    #self.clf.train_data.loc[(DataFrame, label)] = selector.value
                    # Assign a numeric value to make it easier (colormap currently only supports this mode)
                    self.clf.train_data.loc[(self.DataFrame, label)] = choices.index(selector.value)
                    self.update_label_colormap(self.selection_layer, label, choices.index(selector.value))
                else:
                    # TODO: Give feedback to the user that there is no data for a specific label object in the dataframe provided?
                    pass

        @selector.changed.connect
        def change_choice(choice):
            self.selection_layer.visible=True
            self.viewer.layers.selection.clear()
            self.viewer.layers.selection.add(self.label_layer)

        @save_button.changed.connect
        def save_classifier(event):
            print('Saving classifier')
            self.clf.save()

        @run_button.changed.connect
        def run_classifier(event):
            # TODO: Add Run mode? Fuzzy, Cross-validated, train/test split
            print('Running classifier')
            self.clf.train()
            self.create_label_colormap(self.prediction_layer, self.clf.predict_data, 'predict')
            self.clf.save()
            self.selection_layer.visible=False
            self.prediction_layer.visible=True
            # TODO: Report classifier performance to the user?

        return container


    def update_label_colormap(self, curr_label_layer, label, new_class):
        # TODO: This is still kinda laggy on large dataset. Profile this function => can I speed it up?
        # Is there a way to not send a whole new colormap, but just change the colormap in one place?
        # Check if the label_layer has a colormap property that could be modified
        # Check here: viewer.layers['labels'].colormap
        # If I can figure out how to change just the colormap, maybe that is faster than sending a whole colordict
        self.colordict[label] = self.cmap(new_class/self.nb_classes)
        curr_label_layer.color = self.colordict
        # Directly change just the color of the one object, not replacing the whole colormap
        #curr_label_layer.color[label] = self.cmap(new_class/self.nb_classes)
        # Doesn't do anything. Color doesn't update.
        # How is color updated when a whole colormap is provided?
        # This gets to the _color property. But doesn't update the visible colormap

        # See here for discussion on this topic: https://forum.image.sc/t/napari-layer-colormaps-update-individual-objects-only/52547
        # And here for the napari issue: https://github.com/napari/napari/issues/2380


    def create_label_colormap(self, curr_label_layer, df, feature):
        site_df = df[df.index.isin([self.DataFrame], level=0)]
        site_df.index = site_df.index.droplevel()
        colors = self.cmap(site_df[feature]/self.nb_classes)
        colordict = dict(zip(site_df.index, colors))
        curr_label_layer.color = colordict
        return colordict
