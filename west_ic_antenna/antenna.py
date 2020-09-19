# -*- coding: utf-8 -*-
"""
WEST ICRH Antenna RF Model

@author: Julien Hillairet

"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import skrf as rf
from tqdm import tqdm_notebook as tqdm  # nice progress bar
import scipy as sp
import os

# #### Default parameters ####
here = os.path.dirname(os.path.abspath(__file__))
S_PARAMS_DIR = here+'/data/Sparameters/'
# NB : bridge and impedance transformer should be defined for the same frequencies
DEFAULT_BRIDGE = S_PARAMS_DIR+'WEST_ICRH_Bridge_30to70MHz.s3p'
DEFAULT_IMPEDANCE_TRANSFORMER = S_PARAMS_DIR+'WEST_ICRH_Transf_Window_PumpHolePMC.s2p'
# antenna front face data are interpolated on bridge's frequencies 
DEFAULT_FRONT_FACE = S_PARAMS_DIR+'front_faces/WEST_ICRH_antenna_front_face_curved_30to70MHz.s4p'

class WestIcrhAntenna():
    def __init__(self, frequency=None, Cs=[50,50,50,50], antenna_s4p_file=None):
        '''
        WEST ICRH Antenna circuit model

        Parameters
        ----------
        frequency : scikit-rf Frequency, optional
            frequency object to build the circuit with. The default is None (S-param file freq).
            
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]. Default is [50,50,50,50] [pF]
            
        antenna_s4p_file: str, optional
            path to the Touchstone file of the antenna front face. Default is None (Vacuum case). 
            NB: the Frequency object should be compatible with this file. 
            
            
        Capacitor names are defined as (view from behind the antenna, ie from torus hall): 
        
                C1  C3
                C2  C4
        
        Voltages are defined the same way:
            
                V1 V3
                V2 V4
            
        Example
        -------
        >>> freq = rf.Frequency(50, 60, 101, unit='MHz')
        >>> Cs = [50, 40, 60, 70]
        >>> west_antenna = WestIcrhAntenna(freq, Cs)
        
        '''
        # default values
        self._frequency = frequency or rf.Network(DEFAULT_BRIDGE).frequency
        self._Cs = Cs
        self.antenna_s4p_file = antenna_s4p_file or DEFAULT_FRONT_FACE
        
        # load networks
        _bridge = rf.Network(DEFAULT_BRIDGE)

        self.bridge = _bridge.interpolate(self.frequency)
        self.bridge_left = self.bridge.copy()
        self.bridge_left.name = 'bridge_left'
        self.bridge_right = self.bridge.copy()
        self.bridge_right.name = 'bridge_right'

        _windows_impedance_transformer = rf.Network(DEFAULT_IMPEDANCE_TRANSFORMER)

        self.windows_impedance_transformer = _windows_impedance_transformer.interpolate(self.frequency)
        self.windows_impedance_transformer_left = self.windows_impedance_transformer.copy()
        self.windows_impedance_transformer_right = self.windows_impedance_transformer.copy()
        self.windows_impedance_transformer_left.name = 'line_left'
        self.windows_impedance_transformer_right.name = 'line_right'

        # additional elements which will be usefull later
        self.port_left = rf.Circuit.Port(self.frequency, 'port_left', z0=self.windows_impedance_transformer_left.z0[:,0])
        self.port_right = rf.Circuit.Port(self.frequency, 'port_right', z0=self.windows_impedance_transformer_right.z0[:,0])

        
        self._antenna = rf.Network(self.antenna_s4p_file)

        self.antenna = self._antenna.interpolate(self.frequency)
        # self.antenna.name = 'antenna'


    def __repr__(self):
        return f'WEST ICRH Antenna: C={self._Cs} pF, {self._frequency}'

    def capa(self, C, R=1e-2, L=29.9, R1=1e-2, C1=25.7, L1=2.4, z0_bridge=None, z0_antenna=None):
        '''
        Equivalent lumped Network model of a WEST ICRH antenna capacitor.
        
        port1 (bridge side)                        port2 (antenna side)

                o-- R1 -- L1 --- R -- L -- C --- L1 -- R1 --o
                              |               |
                              C1             C1
                              |               |
                              gnd            gnd
                      
        Parameters
        ----------
        C : float
            Capacitance in [pF]
        R : float, optional
            series resitance in [Ohm]. The default is 1e-2.
        L : float, optional
            series inductance in [nH]. The default is 29.9.
        R1 : float, optional
            input/output serie resistance in [Ohm]. The default is 1e-2.
        C1 : float, optional
            shunt capacitance in [pF]. The default is 25.7.
        L1 : float, optional
            input/output series inductance in [nH]. The default is 2.4.
        z0_bridge : float, optional
            Bridge side characteristic impedance in [Ohm]. The default is bridge z0.
        z0_antenna : float, optional
            Antenna side charactetistic impedance in [Ohm]. The default is the antenna z0

        Returns
        -------
        capa : scikit-rf Network
            Equivalent lumped WEST capacitor model Network

        '''
        z0_bridge = z0_bridge or self.bridge.z0[:,1]
        z0_antenna = z0_antenna or self.antenna.z0[:,0]
        # dummy transmission line to create lumped components
        # the 50 Ohm characteristic impedance is artifical. However, the R,L,R1,L1,C1 values
        # have been fitted to full-wave solutions using this 50 ohm value, so it should not be modified
        line = rf.media.DefinedGammaZ0(frequency=self.frequency, z0=50)
        
        pre = line.resistor(R1) ** line.inductor(L1*1e-9) ** line.shunt_capacitor(C1*1e-12)
        post= line.shunt_capacitor(C1*1e-12) ** line.resistor(R1) ** line.inductor(L1*1e-9)
        cap = line.resistor(R) ** line.inductor(L*1e-9) ** line.capacitor(C*1e-12)

        capa = pre ** cap ** post
        
        # should we renormalize of not z0 to the Network's z0 they will be connected to?
        # ANSYS Designer seems not doing it and leaves to 50 ohm
        # renormalizing the z0 will lead to decrease the matched capacitances by ~10pF @55MHz
        # In reality, values are closer to 50 pF at 55 MHz
        # capa.z0 = [z0_bridge, z0_antenna]
        return capa

    def _antenna_circuit(self, Cs):
        '''
        Antenna scikit-rf Circuit

        Parameters
        ----------
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]

        Returns
        -------
        circuit: skrf Circuit
            Antenna Circuit

        '''
        C1, C2, C3, C4 = Cs
        # left side 
        capa_C1 = self.capa(C1, z0_bridge=self.bridge_left.z0[0][1], z0_antenna=self.antenna.z0[0][0])
        capa_C1.name = 'C1'
        capa_C2 = self.capa(C2, z0_bridge=self.bridge_left.z0[0][2], z0_antenna=self.antenna.z0[0][2])
        capa_C2.name = 'C2'
        # right side
        capa_C3 = self.capa(C3, z0_bridge=self.bridge_right.z0[0][1], z0_antenna=self.antenna.z0[0][1])
        capa_C3.name = 'C3'
        capa_C4 = self.capa(C4, z0_bridge=self.bridge_right.z0[0][2], z0_antenna=self.antenna.z0[0][3])
        capa_C4.name = 'C4'

        # WARNING !
        # antenna port numbering convention does not follow capa and voltage :
        # view from behind:
        #   port1  port2
        #   port3  port4
        # while for capa and voltage it is:
        #   C1  C3
        #   C2  C4
        connections = [
            [(self.antenna, 0), (capa_C1, 1)],
            [(self.antenna, 1), (capa_C3, 1)],
            [(self.antenna, 2), (capa_C2, 1)],
            [(self.antenna, 3), (capa_C4, 1)],
            [(capa_C1, 0), (self.bridge_left, 1)],
            [(capa_C2, 0), (self.bridge_left, 2)],
            [(capa_C3, 0), (self.bridge_right, 1)],
            [(capa_C4, 0), (self.bridge_right, 2)],
            [(self.bridge_left, 0), (self.windows_impedance_transformer_left, 1)],
            [(self.bridge_right, 0), (self.windows_impedance_transformer_right, 1)],
            [(self.windows_impedance_transformer_left, 0), (self.port_left, 0)],
            [(self.windows_impedance_transformer_right, 0), (self.port_right, 0)]
        ]
        return rf.Circuit(connections)

    @property
    def Cs(self):
        '''
        Antenna capacitance array [C1, C2, C3, C4] in [pF] 
        '''
        return self._Cs

    @Cs.setter
    def Cs(self, Cs):
        '''
        Set antenna capacitance array [C1, C2, C3, C4]

        Parameters
        ----------
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]

        '''
        self._Cs = Cs

    def circuit(self, Cs=None):
        '''
        Build the antenna circuit for a given set of capacitance

        Parameters
        ----------
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]. Default is None (use internal Cs)

        Returns
        -------
        circuit: skrf Circuit
            Antenna Circuit

        '''
        Cs = Cs or self.Cs
        self._circuit = self._antenna_circuit(Cs)
        return self._circuit

    def _optim_fun_one_side(self, C, f_match=55e6, side='left', z_match=29.89+0j):
        '''
        Optimisation function to match one antenna side. 
        
        Parameters
        ----------
        C : list or array
            half-antenna 2 capacitances [Ctop, Cbot] in [pF].
        f_match: float, optional
            match frequency in [Hz]. Default is 55 MHz.
        side : str, optional
            antenna side to match: 'left' or 'right'
        z_match: complex, optional
            antenna feeder characteristic impedance to match on. Default is 30 ohm
        
        Returns
        -------
        r: float
            Residuals of Z - Z_match
            r = (Z_re - np.real(z_match))**2 + (Z_im - np.imag(z_match))**2
        '''
        Ctop, Cbot = C

        if side == 'left':
            Cs = [Ctop, Cbot, 150, 150]
        elif side == 'right':
            Cs = [150, 150, Ctop, Cbot]

        # Create Antenna network for the capacitances Cs
        # # from Network ('classic way')
        # ntw = half_antenna_network(C, Zload=z_load)
        # # from Circuit
        self._antenna_match.Cs = Cs
        ntw = self._antenna_match.circuit(Cs).network

        # retrieve Z and compare to objective
        index_f_match = np.argmin(np.abs(ntw.f - f_match))

        if side == 'left':
            Z_re = ntw.z_re[index_f_match,0,0].squeeze()
            Z_im = ntw.z_im[index_f_match,0,0].squeeze()
        elif side == 'right':
            Z_re = ntw.z_re[index_f_match,1,1].squeeze()
            Z_im = ntw.z_im[index_f_match,1,1].squeeze()

        r = np.array([  # residuals for both real and imaginary parts
            (Z_re - np.real(z_match)),
            (Z_im - np.imag(z_match))
        ])
        r = (Z_re - np.real(z_match))**2 + (Z_im - np.imag(z_match))**2
        return r


    def match_one_side(self, f_match=55e6, solution_number=1, side='left', z_match=29.89+0j):
        '''
        Search best capacitance to match the specified side of the antenna. 
        
        Capacitance of the non-matched side are set to 120 [pF].

        Parameters
        ----------
        f_match: float, optional
            match frequency in [Hz]. Default is 55 MHz.
        solution_number: int, optional
            1 or 2: 1 for C_top > C_lower or 2 for C_top < C_lower
        side : str, optional
            antenna side to match: 'left' or 'right'
        z_match: complex, optional
            antenna feeder characteristic impedance to match on. Default is 30 ohm            

        Returns
        -------
        Cs_match : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF].

        '''
        # creates an antenna circuit for single frequency only to speed-up calculations
        freq_match = rf.Frequency(f_match, f_match, 1, unit='Hz')
        self._antenna_match = WestIcrhAntenna(freq_match, antenna_s4p_file=self.antenna_s4p_file)

        # setup constraint optimization to force finding the requested solution
        sol_sign = +1 if solution_number == 1 else -1
        A = np.array([[1, 0], [0, 1], [sol_sign*-1, sol_sign*1]])
        lb = np.array([12, 12, -np.inf])
        ub = np.array([150, 150, 0])
        const = scipy.optimize.LinearConstraint(A, lb, ub)

        # try finding a solution until it's a physical one.
        success = False
        while success == False:
            # generate a random C sets, centered on 70 +/- 40
            # satisfying the solution condition
            contin = True
            while contin:
                C0 = 70 + (-1 + 2*scipy.random.rand(2))*40
                if C0[0] > C0[1] and solution_number == 1:
                    contin = False
                elif C0[0] < C0[1] and solution_number == 2:
                    contin = False

            sol = sp.optimize.minimize(self._optim_fun_one_side, C0,
                                          args=(f_match, side, z_match),
                                          constraints=const, method='SLSQP',
                                          )
            # test if the solution found is the capacitor range
            success = sol.success
            if np.isclose(sol.x, 150).any() or \
                np.isclose(sol.x, 12).any() or \
                np.isclose(sol.x[0], sol.x[1]):
                success = False
                print('Wrong solution (out of range capacitor) ! Re-doing...')

            print(success, f'solution #{solution_number}:', sol.x)

        if side == 'left':
            Cs = [sol.x[0], sol.x[1], 150, 150]
        elif side == 'right':
            Cs = [150, 150, sol.x[0], sol.x[1]]

        return Cs
    
    def b(self, a, Cs=None):
        '''
        Reflected power-wave from a given input power-wave, defined by b=S x a

        Parameters
        ----------
        a : array
            input power-wave array
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]. Default is None (use internal Cs)

        Returns
        -------
        b : array
            output power-wave array

        '''
        Cs = Cs or self.Cs  # if not passed use internal Cs
        a_left, a_right = a
        # power waves
        _a = np.zeros(self.circuit(Cs).s.shape[1], dtype='complex')
        # left input
        _a[21] = a_left
        # right input
        _a[23] = a_right
        self._b = self._circuit.s @ _a
        return self._b
    
    def _currents(self, power, phase, Cs=None):
        '''
        Currents at the antenna front face ports (after capacitors)
        OLD EVALUATION BEFORE CIRCUIT IMPLEMENTS VOLTAGE AND CURRENTS

        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]. Default is None (use internal Cs)

        Returns
        -------
        Is : complex array (nbf,4)
            Currents at antenna front face ports [I1, I2, I3, I4]

        Example
        -------
        >>> I1, I2, I3, I4 = west_antenna._currents([1, 1], [0, pi])
        
        '''
        Cs = Cs or self.Cs  # if not passed use internal Cs
        a = self.circuit(Cs)._a_external(power, phase)
        
        b = self.b(a, Cs)
        I1 = (b[:,0] - b[:,1])/np.sqrt(self.antenna.z0[:,0])
        I2 = (b[:,4] - b[:,5])/np.sqrt(self.antenna.z0[:,2])
        I3 = (b[:,2] - b[:,3])/np.sqrt(self.antenna.z0[:,1])
        I4 = (b[:,6] - b[:,7])/np.sqrt(self.antenna.z0[:,3])        
        return np.c_[I1, I2, I3, I4]
    
    def _voltages(self, power, phase, Cs=None):
        '''
        Voltages at the antenna front face ports (after capacitors)
        OLD EVALUATION BEFORE CIRCUIT IMPLEMENTS VOLTAGE AND CURRENTS

        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]. Default is None (use internal Cs)

        Returns
        -------
        Vs : complex array (nbf,4)
            Voltages at antenna front face ports [V1, V2, V3, V4]

        Example
        -------
        >>> V1, V2, V3, V4 = west_antenna._voltages([1, 1], [0, pi])
        
        '''
        Cs = Cs or self.Cs  # if not passed use internal Cs
        a = self.circuit(Cs)._a_external(power, phase)
        
        b = self.b(a, Cs)
        V1 = (b[:,0] + b[:,1])*np.sqrt(self.antenna.z0[:,0])
        V2 = (b[:,4] + b[:,5])*np.sqrt(self.antenna.z0[:,2])
        V3 = (b[:,2] + b[:,3])*np.sqrt(self.antenna.z0[:,1])
        V4 = (b[:,6] + b[:,7])*np.sqrt(self.antenna.z0[:,3])        
        return np.c_[V1, V2, V3, V4]    
    
    @property
    def frequency(self):
        '''
        Antenna Frequency band

        Returns
        -------
        frequency : scikit-rf Frequency
            Antenna Frequency band
        '''
        return self._frequency
    
    @property
    def f(self):
        '''
        Antenna Frequency band values

        Returns
        -------
        f : array
            Antenna Frequency band values in [Hz]
        '''        
        return self._frequency.f
    
    def s_act(self, power, phase, Cs=None):
        '''
        Active S-parameters for a given excitation

        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]
                
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]. Default is None (use internal Cs)
        
        Returns
        -------
        s_act : complex array
            Active S-parameters
        '''
        Cs = Cs or self.Cs
        a = self.circuit(Cs)._a_external(power, phase)
        return self.circuit(Cs).s_active(a)

    def z_act(self, power, phase, Cs=None):
        '''
        Active Z-parameters for a given excitation

        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]. Default is None (use internal Cs)
        
        Returns
        -------
        z_act : complex array
            Active Z-parameters
        '''
        Cs = Cs or self.Cs
        a = self.circuit(Cs)._a_external(power, phase)
        return self.circuit(Cs).z_active(a)
    
    def vswr_act(self, power, phase, Cs=None):
        """
        Active VSWR for a given excitation
        
        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]. Default is None (use internal Cs)
        
        Returns
        -------
        vswr_act : complex array
            Active VSWR-parameters

        """
        Cs = Cs or self.Cs
        s_act = self.s_act(power, phase, Cs)
        
        vswr_left = (1 + np.abs(s_act[:,0]))/(1 - np.abs(s_act[:,0]))
        vswr_right = (1 + np.abs(s_act[:,1]))/(1 - np.abs(s_act[:,1]))
        
        return np.c_[vswr_left, vswr_right]
        
    def voltages(self, power, phase, Cs=None):
        """
        Voltages at the antenna front face ports (after capacitors)

        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]. Default is None (use internal Cs)

        Returns
        -------
        Vs : complex array (nbf,4)
            Voltages at antenna front face ports [V1, V2, V3, V4]

        Example
        -------
        >>> V1, V2, V3, V4 = west_antenna.voltages([1, 1], [0, pi])
        """
        _Cs = Cs or self.Cs
        idx_antenna = [0, 4, 2, 6]  # for port 1,2,3,4 of the antenna
        return self.circuit(_Cs).voltages(power, phase)[:, idx_antenna]
        
    def currents(self, power, phase, Cs=None):
        """
        Currents at the antenna front face ports (after capacitors)

        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]. Default is None (use internal Cs)

        Returns
        -------
        Is : complex array (nbf,4)
            Currents at antenna front face ports [I1, I2, I3, I4]

        Example
        -------
        >>> I1, I2, I3, I4 = west_antenna.currents([1, 1], [0, pi])

        """
        _Cs = Cs or self.Cs
        idx_antenna = [0, 4, 2, 6]  # for port 1,2,3,4 of the antenna
        return self.circuit(_Cs).currents(power, phase)[:, idx_antenna]

    def _Xs(self):
        """
        Xs from interpolation (from Walid)
                    
        Return
        ----------
        - Xs : array
            Strap Admittance best fit

        """
        f_MHz = self._frequency.f/1e6
        p1Xs = 0.000102
        p2Xs = -0.007769
        p3Xs = 0.724653
        p4Xs = -3.175984
        Xs   = p1Xs*f_MHz**3 + p2Xs*f_MHz**2 + p3Xs*f_MHz**1 + p4Xs
        return Xs
    
    def Pr(self, power, phase, Cs=None):
        """
        Reflected power at antenna input.

        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]. Default is None (use internal Cs)        
        

        Returns
        -------
        Pr: complex array (nb_f, 2)
            Reflected power [W] of both sides vs freq
            
        """
        _Cs = Cs or self.Cs
        s_act = self.s_act(power, phase, _Cs)
        return power*np.abs(s_act)**2
        
               
    
    def Rc(self, power, phase, Cs=None):
        """
        Coupling Resistances of both sides of the antenna
        
        Rc = 2Pt/Is^2 where Pt is the transmitted power and Is the average current
        
        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]. Default is None (use internal Cs)        
        

        Returns
        -------
        Rc: array (nb_f, 2)
            Coupling resistances [Ohm] of both sides vs freq

        """
        _Cs = Cs or self.Cs
        # average currents 
        Is = self.currents(power, phase, Cs=_Cs)
        Is_left = np.abs(np.sqrt(Is[:,0]**2 + Is[:,2]**2))
        Is_right =  np.abs(np.sqrt(Is[:,1]**2 +Is[:,3]**2))
        
        # coupled power
        Pr = self.Pr(power, phase, Cs=_Cs)
        Pi = power
        Pt = Pi - Pr
        # coupling resistance 
        Rc_left = 2*Pt[:,0]/Is_left**2
        Rc_right = 2*Pt[:,1]/Is_right**2
        
        return np.c_[Rc_left, Rc_right]
        

    def front_face_Rc(self, Is=[+1,-1,-1,+1]):
        """
        (Ideal) front-face coupling resistance
        
        Parameters
        ----------
        Is : list or array (complex)
            Current excitation at antenna front-face ports. 
            Default is [+1,-1,-1,+1]
        
        """
        
        V = self.antenna.z @ Is
        Prf = 0.5 * V @ Is
        
        I_left_avg = np.sqrt(np.sum(np.abs(Is[0:2])**2)/2)
        I_right_avg = np.sqrt(np.sum(np.abs(Is[2:4])**2)/2)
        
        Rc_left = 2*Prf.real/I_left_avg**2
        Rc_right = 2*Prf.real/I_right_avg**2
        return np.c_[Rc_left, Rc_right]