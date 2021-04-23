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

class ColormapChoices(Enum):
    viridis='viridis'
    test='test'

@magic_factory(
        call_button="Apply Feature Colormap",
        layout='vertical',
        DataFrame={'mode': 'r'},
        upper_constrast_limit={"min": -100000, "max": 100000}
        )
def feature_vis(label_layer: "napari.layers.Labels",
                DataFrame: pathlib.Path,
                feature = 'feature1',
                label_column = 'label',
                Colormap=ColormapChoices.viridis,
                auto_detect_contrast_limits=True,
                lower_contrast_limit: int = 100, upper_constrast_limit: int = 900):
    # TODO: handle the colormap choice
    site_df = pd.read_csv(DataFrame)
    site_df.loc[:, 'label'] = site_df[str(label_column)].astype(int)
    # Check that there is one unique label for every entry in the dataframe
    # => It's a site dataframe, not one containing many different sites
    assert len(site_df['label'].unique()) == len(site_df), 'A feature dataframe with non-unique labels was provided. The visualize_feature_on_label_layer function is not designed for this.'
    # Rescale feature between 0 & 1 to make a colormap
    # If a threshold tuple is provided, the first value is used as lower and the second value as upper boundary
    if auto_detect_contrast_limits:
        quantiles=(0.01, 0.99)
        lower_contrast_limit = site_df[feature].quantile(quantiles[0])
        upper_constrast_limit = site_df[feature].quantile(quantiles[1])
        
    site_df['feature_scaled'] = (
        (site_df[feature] - lower_contrast_limit) / (upper_constrast_limit - lower_contrast_limit)
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
    #label_layer.properties = label_properties

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    #return [ExampleQWidget, example_magic_widget, feature_vis]
    return feature_vis
