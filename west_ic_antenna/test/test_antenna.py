# -*- coding: utf-8 -*-
import pytest
import skrf as rf
import os

from west_ic_antenna.antenna import (
    WestIcrhAntenna,
    DEFAULT_FRONT_FACE,
    S_PARAMS_DIR,
    DEFAULT_BRIDGE,
    DEFAULT_IMPEDANCE_TRANSFORMER,
    DEFAULT_SERVICE_STUB,
)

# Useful definitions
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TEST_DIR, '../data')

@pytest.fixture
def antenna_default_arg():
    return WestIcrhAntenna()

# Constructor Tests
def test_constructor_default(antenna_default_arg):
    """Test creation of Antenna instance"""
    assert isinstance(antenna_default_arg, WestIcrhAntenna)
    
def test_setup_constructor_frequency():
    freq = rf.Frequency(30, 60, npoints=10, unit='MHz')
    _ant = WestIcrhAntenna(frequency=freq)
    assert isinstance(_ant, WestIcrhAntenna)
    
def test_setup_constructor_capa():
    Cs = [20, 30, 40, 50]
    _ant = WestIcrhAntenna(Cs=Cs)        
    assert isinstance(_ant, WestIcrhAntenna)
    
def test_setup_constructor_frontface_filename():
    _ant = WestIcrhAntenna(front_face=DEFAULT_FRONT_FACE)        
    assert isinstance(_ant, WestIcrhAntenna)        

def test_setup_constructor_frontface_network():
    ntw = rf.Network(DEFAULT_FRONT_FACE)
    _ant = WestIcrhAntenna(front_face=ntw)        
    assert isinstance(_ant, WestIcrhAntenna)     

