from collections import OrderedDict
from zlib import crc32
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from magicgui import magic_factory
from .utils import get_df
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

#DataFrame = '/Users/joel/Dropbox/Joel/PelkmansLab/Code/napari-feature-visualization/napari_feature_visualization/test_df_3.csv',
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
    clf.save()

    # Create a selection & prediction layer
    # TODO: Handle state when those layers were already created. Replace them otherwise?
    # https://napari.org/guides/stable/magicgui.html#updating-an-existing-layer
    selection_layer = viewer.add_labels(label_layer.data, name='selection', opacity=1.0)
    prediction_layer = viewer.add_labels(label_layer.data, name='prediction', opacity=1.0)
    update_label_colormap(selection_layer, clf.train_data, 'train', DataFrame)
    update_label_colormap(prediction_layer, clf.predict_data, 'predict', DataFrame)
    viewer.layers.selection.clear()
    viewer.layers.selection.add(label_layer)

    widget = selector_widget(clf, label_layer, DataFrame, selection_layer, prediction_layer, viewer)

    # TODO: Add a warning if a classifier with this name already exists => shall it be overwritten? => Confirmation box

    # add widget to napari
    viewer.window.add_dock_widget(widget, area='right', name=classifier_name)


@magic_factory(
        call_button="Load Classifier",
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
    classifier_name = classifier_path.stem

    with open(classifier_path, 'rb') as f:
        clf = pickle.loads(f.read())

    training_features = clf.training_features
    site_df = get_df(DataFrame)
    site_df['path']=DataFrame
    index_columns=clf.index_columns
    # Catches if new data frame doesn't contain the index columns
    assert(all([index_column in site_df.columns for index_column in index_columns]))
    # TODO: Notify the user why the classifier is not loaded
    site_df = site_df.set_index(list(index_columns))

    # TODO: Check if data needs to be added to the classifier
    clf.add_data(site_df, training_features=training_features, index_columns=index_columns)
    clf.save()

    # Create a selection & prediction layer
    # TODO: Handle state when those layers were already created. Replace them otherwise?
    # https://napari.org/guides/stable/magicgui.html#updating-an-existing-layer
    selection_layer = viewer.add_labels(label_layer.data, name='selection', opacity=1.0)
    prediction_layer = viewer.add_labels(label_layer.data, name='prediction', opacity=1.0)
    update_label_colormap(selection_layer, clf.train_data, 'train', DataFrame)
    update_label_colormap(prediction_layer, clf.predict_data, 'predict', DataFrame)
    viewer.layers.selection.clear()
    viewer.layers.selection.add(label_layer)

    widget = selector_widget(clf, label_layer, DataFrame, selection_layer, prediction_layer, viewer)

    # add widget to napari
    viewer.window.add_dock_widget(widget, area='right', name=classifier_name)


def selector_widget(clf, label_layer, DataFrame, selection_layer, prediction_layer, viewer):
    # TODO: Define a minimum number of selections. Below that, show a warning (training-test split can be very weird otherwise, e.g all 1 class)
    # TODO: Generalize this. Instead of 0, 1, 2: Arbitrary class numbers. Ability to add classes & name them?
    choices = ['Deselect', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    selector = widgets.RadioButtons(choices=choices, label='Selection Class:')
    save_button = widgets.PushButton(value=True, text='Save Classifier')
    run_button = widgets.PushButton(value=True, text='Run Classifier')
    container = widgets.Container(widgets=[selector, save_button, run_button])
    # TODO: Add text field & button to save classifier output to disk

    @label_layer.mouse_drag_callbacks.append
    def toggle_label(label_layer, event):
        label = label_layer.get_value(event.position)
        # Check if background or foreground was clicked. If background was clicked, do nothing (background can't be assigned a class)
        if label == 0:
            pass
        else:
            # Check if the label exists in the current dataframe. Otherwise, do nothing
            if (DataFrame, label) in clf.train_data.index:
                # Assign name of class
                #clf.train_data.loc[(DataFrame, label)] = selector.value
                # Assign a numeric value to make it easier (colormap currently only supports this mode)
                clf.train_data.loc[(DataFrame, label)] = choices.index(selector.value)
                update_label_colormap(selection_layer, clf.train_data, 'train', DataFrame)
            else:
                # TODO: Give feedback to the user that there is no data for a specific label object in the dataframe provided?
                pass

    @selector.changed.connect
    def change_choice(choice):
        selection_layer.visible=True
        prediction_layer.visible=False
        viewer.layers.selection.clear()
        viewer.layers.selection.add(label_layer)

    @save_button.changed.connect
    def save_classifier(event):
        print('Saving classifier')
        clf.save()

    @run_button.changed.connect
    def run_classifier(event):
        print('Running classifier')
        clf.train()
        update_label_colormap(prediction_layer, clf.predict_data, 'predict', DataFrame)
        clf.save()

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
    label_layer.visible=True



##################################
# Actual classification code
##################################

'''
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
'''

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

    # TODO: Change back test_perc to something more reasonable like 0.2
    # Having it at 0.3 now to reduce issues when the dataset has no test samples
    @staticmethod
    def train_test_split(df, test_perc=0.3, index_columns=None):
        # TODO: Ensure at least 1 per class?
        in_test_set = make_identifier(df.reset_index()[list(index_columns)]).apply(
            test_set_check, args=(test_perc,)
        )
        return df.iloc[~in_test_set.values, :], df.iloc[in_test_set.values, :]

    #TODO: Add a add_data method that checks if the data is already in the
    #classifier and adds it otherwise
    def add_data(self, features, training_features, index_columns):
        # Check that training features agree with already existing training features
        assert training_features == self.training_features
        # Optionally: Allow option to change training features. Would mean:
        # Classifier also needs to know the paths to all the full data added before so it can reload that
        # And then go through all existing data to load extra features => separate method to be written first

        # Check if data with the same index already exists. If so, do nothing
        assert index_columns == self.index_columns, 'The newly added dataframe \
                                                    uses different index columns \
                                                    than what was used in the \
                                                    classifier before: New {}, \
                                                    before {}'.format(index_columns,
                                                                      self.index_columns)
        # Check which indices already exist in the data, only add the others
        new_indices = self._index_not_in_other_df(features, self.train_data)
        new_data = features.loc[new_indices['index_new']]
        if len(new_data.index) == 0:
            # No new data to be added: The classifier is being loaded for a
            # site where the data has been loaded before
            pass
        else:
            new_data.loc['train'] = 0
            new_data.loc['predict'] = 0
            self.train_data = self.train_data.append(new_data[['train']])
            self.predict_data = self.predict_data.append(new_data[['predict']])
            self.data = self.data.append(new_data[training_features])


    @staticmethod
    def _index_not_in_other_df(df1, df2):
        # Function checks which indices of df1 already exist in the indices of df2.
        # Returns a boolean pd.DataFrame with a 'index_preexists' column
        df_overlap = pd.DataFrame(index=df1.index)
        for df1_index in df1.index:
            if df1_index in df2.index:
                df_overlap.loc[df1_index, 'index_new'] = False
            else:
                df_overlap.loc[df1_index, 'index_new'] = True
        return df_overlap

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
