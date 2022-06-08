# -*- coding: utf-8 -*-
import pathlib


def main(data: list[pathlib.Path], destination: pathlib.Path):
    destination.mkdir(parents=True, exist_ok=True)
    print("Coefficients destination:", destination)

    print("Saving coefficients:")
    for dataset in data:
        print("\tloaded", dataset.absolute())
