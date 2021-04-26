"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from magicgui import magic_factory
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from functools import lru_cache


class ColormapChoices(Enum):
    viridis='viridis'
    test='test'


@lru_cache(maxsize=16)
def get_df(path):
    return pd.read_csv(path)


def _init(widget):
    @widget.DataFrame.changed.connect
    def update_df_columns(event):
        # Implemented following inputs from Talley Lambert:
        # https://forum.image.sc/t/visualizing-feature-measurements-in-napari-using-colormaps-as-luts/51567/16
        # event value will be the new path
        # get_df will give you the cached df
        df = get_df(event.value)
        features = list(df.columns)
        widget.feature.choices = features
        widget.label_column.choices = features
        if 'label' in features:
            widget.label_column.value = 'label'
        elif 'Label' in features:
            widget.label_column.value = 'Label'
        elif 'index' in features:
            widget.label_column.value = 'index'


    @widget.feature.changed.connect
    def update_rescaling(event):
        df = get_df(widget.DataFrame.value)
        # TODO: Check whether feature is in df
        try:
            quantiles=(0.01, 0.99)
            widget.lower_contrast_limit.value = df[event.value].quantile(quantiles[0])
            widget.upper_contrast_limit.value = df[event.value].quantile(quantiles[1])
        except KeyError:
            # Don't update the limits if a feature name is entered that isn't in the dataframe
            pass



# TODO: Set better limits for contrast_limits
@magic_factory(
        call_button="Apply Feature Colormap",
        layout='vertical',
        DataFrame={'mode': 'r'},
        lower_contrast_limit={"min": -1000000, "max": 1000000},
        upper_contrast_limit={"min": -1000000, "max": 1000000},
        feature = {"choices": [""]},
        label_column = {"choices": [""]}, widget_init=_init,
        )
def feature_vis(label_layer: "napari.layers.Labels",
                DataFrame: pathlib.Path,
                feature = '',
                label_column = '',
                Colormap=ColormapChoices.viridis,
                lower_contrast_limit: int = 100, upper_contrast_limit: int = 900):
    # TODO: handle the colormap choice
    site_df = get_df(DataFrame)
    site_df.loc[:, 'label'] = site_df[str(label_column)].astype(int)
    # Check that there is one unique label for every entry in the dataframe
    # => It's a site dataframe, not one containing many different sites
    # TODO: How to feedback this issue to the user?
    assert len(site_df['label'].unique()) == len(site_df), 'A feature dataframe with non-unique labels was provided. The visualize_feature_on_label_layer function is not designed for this.'
    # Rescale feature between 0 & 1 to make a colormap
    site_df['feature_scaled'] = (
        (site_df[feature] - lower_contrast_limit) / (upper_contrast_limit - lower_contrast_limit)
    )
    # Cap the measurement between 0 & 1
    site_df.loc[site_df['feature_scaled'] < 0, 'feature_scaled'] = 0
    site_df.loc[site_df['feature_scaled'] > 1, 'feature_scaled'] = 1

    # TODO: Handle different colormaps
    colors = plt.cm.viridis(site_df['feature_scaled'])

    # Create an array where the index is the label value and the value is
    # the feature value
    properties_array = np.zeros(site_df['label'].max() + 1)
    properties_array[site_df['label']] = site_df[feature]
    label_properties = {feature: np.round(properties_array, decimals=2)}

    colormap = dict(zip(site_df['label'], colors))
    label_layer.color = colormap
    # TODO: Wait for fix for https://github.com/napari/napari/issues/2477
    # Once this is in release, add this to the plugin
    #label_layer.properties = label_properties


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    #return [ExampleQWidget, example_magic_widget, feature_vis]
    return feature_vis
