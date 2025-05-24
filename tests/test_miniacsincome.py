# Copyright (c) 2025 David Boetius
# Licensed under the MIT license
from miniacsincome import MiniACSIncome, get_network, get_population_model


def test_load_dataset():
    print(MiniACSIncome(download=True))

def test_get_network():
    print(get_network(3, download=True))

def test_get_population_model():
    print(get_population_model(3, download=True))
