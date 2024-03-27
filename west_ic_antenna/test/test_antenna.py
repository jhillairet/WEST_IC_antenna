# -*- coding: utf-8 -*-
import os
import pytest
import skrf as rf
import numpy as np


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
DATA_DIR = os.path.join(TEST_DIR, "../data")


@pytest.fixture
def antenna_default_arg():
    return WestIcrhAntenna()


# Constructor Tests
def test_constructor_default(antenna_default_arg):
    """Test creation of Antenna instance"""
    assert isinstance(antenna_default_arg, WestIcrhAntenna)


def test_setup_constructor_frequency():
    freq = rf.Frequency(30, 60, npoints=10, unit="MHz")
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


# Test capacitor model
"""
Equivalent lumped Network model of a WEST ICRH antenna capacitor.

The electrical circuit of an equivalent lumped model is:

    port1 (bridge side)                        port2 (antenna side)

        o-- R1 -- L1 --- R -- L -- C --- L1 -- R1 --o
                      |               |
                      C1             C1
                      |               |
                      gnd            gnd
"""


def test_capa_model_50ohm_ports(antenna_default_arg):
    """
    Benchmark capacitor model with ANSYS Circuit model.

    Default values and 50 Ohm port impedance in ANSYS.
    """
    ant = antenna_default_arg
    cap = ant.capa(
        C=50,
        z0_bridge=50,
        z0_antenna=50,
    )

    # Does it returns a Network?
    assert isinstance(cap, rf.Network)

    # ANSYS Equivalent circuit with 50 Ohm ports
    ANSYS_model = (
        "ANSYS_benchmarks/WEST_capacitor_equivalent_circuit_50ohm_ports.s2p"
    )
    cap_ANSYS = rf.Network(os.path.join(TEST_DIR, ANSYS_model))

    assert cap == cap_ANSYS


def test_capa_model_bridge_antenna_ports(antenna_default_arg):
    """
    Benchmark capacitor model with ANSYS Circuit model.

    Specifiyng bridge and antenna ports'Z0 in ANSYS (real values).
    """
    ant = antenna_default_arg
    cap = ant.capa(C=50)

    # ANSYS Equivalent circuit with complex bridge and antenna ports' Z0
    ANSYS_model = "ANSYS_benchmarks/WEST_capacitor_equivalent_circuit_bridge_antenna_ports.s2p"
    cap_ANSYS = rf.Network(os.path.join(TEST_DIR, ANSYS_model))

    assert cap == cap_ANSYS


def test_capa_model_bridge_antenna_ports_renorm50ohm(antenna_default_arg):
    """
    Benchmark capacitor model with ANSYS Circuit model.

    Specifiyng bridge and antenna ports'Z0 in ANSYS (real values),
    but renormalizing ANSYS results to 50 ohm during export of Touchstone.
    """
    ant = antenna_default_arg
    cap_50 = ant.capa(C=50, z0_bridge=50, z0_antenna=50)

    ANSYS_model_50 = "ANSYS_benchmarks/WEST_capacitor_equivalent_circuit_bridge_antenna_ports_renorm50ohm.s2p"
    cap_ANSYS_50 = rf.Network(os.path.join(TEST_DIR, ANSYS_model_50))

    assert cap_50 == cap_ANSYS_50


def test_capa_model_connection(antenna_default_arg):
    """
    Benchmark dummy circuit model of a antenna-capa-bridge vs ANSYS Circuit.

    In ANSYS, a single capacitor electrical circuit is connected to one bridge
    output and to an antenne front face input (using HFSS model).
    Bridge input is connected to input port and all other ports are shorted.

    Benchmark is performed when the Touchtone file is exported without and with
    50 Ohm renormalization.

    """
    ant = antenna_default_arg
    cap = ant.capa(C=50)

    # Creating the dummy circuit using scikit-rf
    cap.name = 'capa'
    port = rf.Circuit.Port(frequency=ant.frequency, name='port1', z0=ant.bridge.z0[:,0].real)
    gnd = rf.Circuit.Ground(frequency=ant.frequency, name="gnd")
    cnx = [
        [(ant.bridge, 0), (port, 0)],
        [(ant.bridge, 1), (cap, 0)],
        [(cap, 1), (ant.antenna, 0)],
        # grounding all other ports
        [(gnd, 0), (ant.bridge, 2), (ant.antenna, 1), (ant.antenna, 2), (ant.antenna, 3)]
    ]
    cir = rf.Circuit(cnx)
    ntw = cir.network

    # ANSYS Model Export without renormalization
    ANSYS_model = "ANSYS_benchmarks/WEST_capacitor_test_connection.s1p"
    ANSYS_connection_test = rf.Network(os.path.join(TEST_DIR, ANSYS_model))

    assert ntw == ANSYS_connection_test

    # ANSYS Model Export Renomalized to 50 Ohm
    ANSYS_model_50 = "ANSYS_benchmarks/WEST_capacitor_test_connection_renorm50ohm.s1p"
    ANSYS_connection_test_50 = rf.Network(os.path.join(TEST_DIR, ANSYS_model_50))
    ntw.renormalize(50)

    assert ntw == ANSYS_connection_test_50

# Various return value tests
def test_antenna_circuit(antenna_default_arg):
    """Test the internal circuit."""
    ant = antenna_default_arg
    assert isinstance(ant.circuit(), rf.Circuit)
    assert isinstance(ant.circuit(Cs=[50,50,50,50]), rf.Circuit)

    # TODO benchmark against ANSYS for a specific case of capa set

def test_Cs(antenna_default_arg):
    ant = antenna_default_arg
    assert isinstance(ant.Cs, list)


# Matching Tests (longer...)
def test_match_one_side_left(antenna_default_arg):
    # Test the return values of left side matching
    Cs = antenna_default_arg.match_one_side(
        f_match=55e6,
        solution_number=1,
        side="left",
        z_match=29.89 + 0j
        )
    # reference values 25/03/2024
    np.testing.assert_allclose(Cs, [51.44282751806046, 49.325275021316564, 150, 150], rtol=1e-5)

def test_match_one_side_right(antenna_default_arg):
    # Test the return values of right side matching
    Cs = antenna_default_arg.match_one_side(
        f_match=55e6,
        solution_number=1,
        side="right",
        z_match=29.89 + 0j
        )
    # reference values 25/03/2024
    np.testing.assert_allclose(Cs, [150, 150, 51.20960703555699, 49.489644871023714], rtol=1e-5)

def test_match_both_sides_separately(antenna_default_arg):
    # Test the return values of both sides matching (separately)
    Cs = antenna_default_arg.match_both_sides_separately(
        f_match=55e6,
        solution_number=1,
        z_match=[29.89 + 0j, 29.87 + 0j],
        )
    # reference values 25/03/2024
    np.testing.assert_allclose(Cs, [51.44282792582701, 49.32527088864287, 51.20987339854108, 49.48946981138618], rtol=1e-1)

def test_match_both_sides(antenna_default_arg):
    # Test the return values of both sides matching
    Cs = antenna_default_arg.match_both_sides(
        f_match=55e6,
        solution_number=1,
        z_match=[29.89 + 0j, 29.87 + 0j],
        )
    # reference values 25/03/2024
    np.testing.assert_allclose(Cs, [52.09695694, 50.08075395, 51.62129529, 50.25710599], rtol=1e-1)


if __name__ == "__main__":
    pytest.main([__file__])
