#!/usr/bin/env bash

# pytest --cov-report term-missing --cov=symbols symbols

pytest --cov-report html --cov=symbols symbols
# xdg-open htmlcov/index.html