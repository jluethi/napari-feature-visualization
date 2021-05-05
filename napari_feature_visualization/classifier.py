from collections import OrderedDict
from zlib import crc32
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from magicgui import magic_factory
#import magicgui
from .utils import get_df
#from napari.layers import Labels
#from napari.types import LabelsData, ImageData
from napari import Viewer
from magicgui import widgets


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

    @widget.DataFrame.changed.connect
    def update_df_columns(event):
        # event value will be the new path
        # get_df will give you the cached df
        # ...reset_choices() calls the "get_feature_choices" function above
        # to keep them updated with the current dataframe
        widget.feature_selection.reset_choices()

#DataFrame: Path,
@magic_factory(
        call_button="Initialize Classifier",
        feature_selection = {"choices": [""]}, widget_init=_init_classifier,
        )
def classifier_initialize(viewer: Viewer,
                      label_layer: "napari.layers.Labels",
                      DataFrame = '/Users/joel/Dropbox/Joel/PelkmansLab/Code/napari-feature-visualization/napari_feature_visualization/test_df_3.csv',
                      classifier_name = 'test',
                      feature_selection=''):
    # TODO: Add option to load a classifier. Potentially separate widget, e.g. widget 1 = create classifier. Widget 2 = load classifier
    # TODO: Make feature selection a widget that allows multiple features to be selected, not just one
    # TODO: Make the label column in index something selectable
    training_features = [feature_selection]

    site_df = get_df(DataFrame)
    site_df['path']=DataFrame
    index_columns=('path', 'label')
    site_df = site_df.set_index(list(index_columns))
    clf = Classifier(name=classifier_name, features=site_df, training_features=training_features, index_columns=index_columns)
    clf.save()

    # Create a selection & prediction layer
    # TODO: Handle state when those layers were already created. Replace them otherwise?
    # TODO: Use the clf data to set the colormap, based on classifier.train_data (all 0s to start with)
    # https://napari.org/guides/stable/magicgui.html#updating-an-existing-layer
    selection_layer = viewer.add_labels(label_layer.data, name='selection', opacity=1.0)
    prediction_layer = viewer.add_labels(label_layer.data, name='prediction', opacity=1.0)
    update_label_colormap(selection_layer, clf.train_data, 'train', DataFrame)
    update_label_colormap(prediction_layer, clf.predict_data, 'predict', DataFrame)
    viewer.layers.selection.clear()
    viewer.layers.selection.add(label_layer)
    #label_layer.selected = True
    #selection_layer.selected = False
    #prediction_layer.selected = False

    widget = selector_widget(clf, label_layer, DataFrame, selection_layer)

    # add widget to napari
    viewer.window.add_dock_widget(widget, area='right', name=classifier_name)


def selector_widget(clf, label_layer, DataFrame, selection_layer):
    # TODO: Generalize this. Instead of 0, 1, 2: Arbitrary class numbers. Ability to add classes & name them?
    choices = ['Deselect', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    selector = widgets.RadioButtons(choices=choices, label='Selection Class:')
    save_button = widgets.PushButton(value=True, text='Save Classifier')
    run_button = widgets.PushButton(value=True, text='Run Classifier')
    container = widgets.Container(widgets=[selector, save_button, run_button])

    @label_layer.mouse_drag_callbacks.append
    def toggle_label(label_layer, event):
        label = label_layer.get_value(event.position)
        #clf.train_data.loc[(DataFrame, label)] = selector.value
        # Assign a numeric value to make it easier
        clf.train_data.loc[(DataFrame, label)] = choices.index(selector.value)
        # TODO: Update colormap => write function for this
        update_label_colormap(selection_layer, clf.train_data, 'train', DataFrame)


    @selector.changed.connect
    def change_choice(choice):
        viewer.layers.selection.clear()
        viewer.layers.selection.add(label_layer)

    @save_button.changed.connect
    def save_classifier(event):
        print('Saving classifier')
        clf.save()

    @run_button.changed.connect
    def run_classifier(event):
        print('Running classifier')
        # TODO: Trigger classifier run

    return container


def update_label_colormap(label_layer, df, feature, DataFrame):
    # TODO: Implement a real colormap for this
    # Currently doesn't work for more than 5 classes (0-4)
    manual_cmap = np.array([(0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0),
                            (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)])

    color_dict = {}
    for label in df.index.get_level_values(1):
        color_dict[label] = manual_cmap[df.loc[(DataFrame, label), feature]]
    label_layer.color = color_dict



##################################
# Actual classification code
##################################

def load_features(fld, structure=None, index_col=("filename_prefix", "Label"), glob_str="*.csv"):
    fld = Path(fld)
    features = []
    for fn in fld.glob(glob_str):
        print(fn.name)
        features.append(pd.read_csv(fn))
    df = pd.concat(features, axis=0, ignore_index=True)
    if structure:
        df = df[df["structure"] == structure]
    df = df.set_index(list(index_col))
    return df


def read_feather(fn, index_col=("filename_prefix", "Label")):
    df = pd.read_feather(fn)
    if index_col:
        df = df.set_index(list(index_col))
    return df


def make_identifier(df):
    str_id = df.apply(lambda x: "_".join(map(str, x)), axis=1)
    return str_id


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(hash(identifier))) & 0xFFFFFFFF < test_ratio * 2 ** 32


class Classifier:
    def __init__(self, name, features, training_features, index_columns=None):
        self.name = name
        self.clf = RandomForestClassifier()
        full_data = features
        full_data.loc[:, "train"] = 0
        full_data.loc[:, "predict"] = 0
        self.index_columns = index_columns
        self.train_data = full_data[["train"]]
        self.predict_data = full_data[["predict"]]
        self.training_features = training_features
        self.data = full_data[self.training_features]

    @staticmethod
    def train_test_split(df, test_perc=0.2, index_columns=None):
        in_test_set = make_identifier(df.reset_index()[list(index_columns)]).apply(
            test_set_check, args=(test_perc,)
        )
        return df.iloc[~in_test_set.values, :], df.iloc[in_test_set.values, :]

    #TODO: Add a add_data method that checks if the data is already in the
    #classifier and adds it otherwise
    def add_data(self, features, training_features):
        # Check that training features agree with already existing training features
        # Optionally: Allow option to change training features. Would mean:
        # Classifier also needs to know the paths to all the full data added before so it can reload that
        raise NotImplementedError

    def train(self):
        X_train, X_test = self.train_test_split(
            self.data[self.train_data["train"] > 0], index_columns=self.index_columns
        )
        y_train, y_test = self.train_test_split(
            self.train_data[self.train_data["train"] > 0], index_columns=self.index_columns
        )
        assert np.all(X_train.index == y_train.index)
        assert np.all(X_test.index == y_test.index)
        print(
            "Annotations split into {} training and {} test samples...".format(
                len(X_train), len(X_test)
            )
        )
        self.clf.fit(X_train, y_train)

        print(
            "F1 score on test set: {}".format(
                f1_score(y_test, self.clf.predict(X_test), average="macro")
            )
        )
        self.predict_data.loc[:] = self.clf.predict(self.data).reshape(-1, 1)
        print("done")

    def predict(self, data):
        data = data[self.training_features]
        return self.clf.predict(data)

    def feature_importance(self):
        return OrderedDict(
            sorted(
                {
                    f: i
                    for f, i in zip(
                    self.training_features, self.clf.feature_importances_
                )
                }.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def most_important(self, n=5):
        return list(self.feature_importance().keys())[:n]

    def save(self):
        s = pickle.dumps(self)
        with open(self.name + ".clf", "wb") as f:
            f.write(s)


if __name__ == "__main__":
    main()
