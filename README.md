
# napari-feature-visualization

[![License](https://img.shields.io/pypi/l/napari-feature-visualization.svg?color=green)](https://github.com/jluethi/napari-feature-visualization/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-feature-visualization.svg?color=green)](https://pypi.org/project/napari-feature-visualization)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-feature-visualization.svg?color=green)](https://python.org)
[![tests](https://github.com/jluethi/napari-feature-visualization/workflows/tests/badge.svg)](https://github.com/jluethi/napari-feature-visualization/actions)
[![codecov](https://codecov.io/gh/jluethi/napari-feature-visualization/branch/master/graph/badge.svg)](https://codecov.io/gh/jluethi/napari-feature-visualization)

Napari plugin to visualize feature measurements on label images

----------------------------------
This plugin currently contains two widgets:
1. Feature visualization  
![napari_plugin_on_actual_data_small](https://user-images.githubusercontent.com/18033446/116708698-40c6e380-a9d0-11eb-8e9f-97a257c7bc33.gif)
![napari-feature-viz](https://user-images.githubusercontent.com/18033446/115883664-54150480-a44e-11eb-93df-ab355bb3db89.gif)

2. Calculation of scikit-image regionprops
![a15ba3b2f55de8679b91bda4f41a2b7b80691292_2_690x390](https://user-images.githubusercontent.com/18033446/117011582-4123ef00-acee-11eb-9c43-bf9336dcb038.jpeg)

Features calculated with the regionprops widget can be saved to csv and visualized using the feature_vis widget.

----------------------------------
This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/docs/plugins/index.html
-->

## Installation

This plugin is not available via pipy yet.
So far, git clone it and pip install it using:
```
    cd napari-feature-visualization
    pip install .
```
For full functionality, recent bug-fixes in magicgui are required that have not been released via pipy yet. To get the pre-release version, install them directly:
```
pip install git+https://github.com/napari/magicgui.git
```

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-feature-visualization" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/jluethi/napari-feature-visualization/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

## Contributors
Feature visualization widget: [Joel Lüthi](https://github.com/jluethi)  
Classifier widgets: [Joel Lüthi](https://github.com/jluethi) & [Max Hess](https://github.com/MaksHess)  
Regionprops widget: [Robert Haase](https://github.com/haesleinhuepf)  

And support by people from the napari & iamge.sc community like Nicholas Sofroniew, Juan Nunez Iglesias, Draga Doncila Pop, Talley Lambert and many more

