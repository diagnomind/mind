# web-server
[![Python application](https://github.com/diagnomind/mind/actions/workflows/python-app.yml/badge.svg)](https://github.com/diagnomind/mind/actions/workflows/python-app.yml)

[![Quality Gate Status](https://sonarqube.diagnomind.duckdns.org/api/project_badges/measure?project=mind&metric=alert_status&token=sqb_e7fe286b645d9b92e0772f578e2373f2b261519b)](https://sonarqube.diagnomind.duckdns.org/dashboard?id=mind)

## Description

This repository contains the AI of Diagnomind. It segments tumor areas from CT scans using CNN with the U-Net architecture. This model can be deployed as REST server, and the notebooks contain the model.

## Installation

To install this project Python 3.11 is recommended. Running on a Linux environment is also strongly recommended, due to the fact that Tensorflow has dropped support for GPU acceleration in Windows machines. Check the [pyproject.toml](pyproject.toml) for information on project dependencies.

With the following command the REST server can be installed with the minimal dependencies:
```
$> pip install .
```

To install with additional development dependencies:
```
$> pip install .[dev]
```

To install with notebook dependencies:
```
$> pip install .[notebook]
```

To install with all the optional dependencies:
```
$> pip install .[all]
```

To build a `.whl` package:
```
$> python -m build
```

To run tests and check coverage:
```
$> coverage run -m --source=mind unittest discover tests    # Will run the tests
$> coverage html    # This will produce an html to see the coverage in every file.
```

Note that the `mind.keras` model must be in `src/mind/model/mind.keras`, to be able to execute correctly afterwards.

## Usage

Once the project is installed just run the `mind` command in the terminal like this:
```
$> mind
```

This will run the web server. 

## Credits

- Qing Yu Jiang Pan
- Diogo Sousa Fernandes
- Gaizka Sáenz de Samaniego Gonzalez
- Asier López Lorenzo
- Eñaut Genua Prieto

## License

This project is licensed under the [AGPLv3+](LICENSE).