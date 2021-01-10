# -*- coding: utf-8 -*-
"""
WEST ICRH Antenna RF Model

@author: Julien Hillairet

"""
import os
import scipy
import skrf as rf
import numpy as np

# #### Default parameters ####
here = os.path.dirname(os.path.abspath(__file__))
S_PARAMS_DIR = here+'/data/Sparameters/'
# NB : bridge and impedance transformer should be defined for the same frequencies
DEFAULT_BRIDGE = S_PARAMS_DIR+'WEST_ICRH_Bridge_30to70MHz.s3p'
DEFAULT_IMPEDANCE_TRANSFORMER = S_PARAMS_DIR+'WEST_ICRH_Transf_Window_PumpHolePMC.s2p'
DEFAULT_SERVICE_STUB = S_PARAMS_DIR+'WEST_ICRH_Stub_30to70MHz.s3p'
# antenna front face data are interpolated on bridge's frequencies
DEFAULT_FRONT_FACE = S_PARAMS_DIR+'front_faces/WEST_ICRH_antenna_front_face_curved_30to70MHz.s4p'
# Optimal Impedance at T-junction
Z_T_OPT = 2.89 - 0.17j

class WestIcrhAntenna():
    def __init__(self, frequency=None, Cs=[50,50,50,50], front_face=None):
        '''
        WEST ICRH Antenna circuit model

        Parameters
        ----------
        frequency : scikit-rf Frequency, optional
            frequency object to build the circuit with. The default is None (S-param file freq).

        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]. Default is [50,50,50,50] [pF]

        front_face: str or skrf.Network, optional
            str: path to the Touchstone file of the antenna front face. Default is None (Vacuum case).
            NB: the Frequency object should be compatible with this file.

        Remarks
        --------
        front face ports are defined as (view from behind, ie from torus hall):

            port1  port2
            port3  port4

        Capacitor names are defined as (view from behind the antenna):

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
        self._frequency = frequency or rf.Network(DEFAULT_BRIDGE).frequency
        self._Cs = Cs
        
        # display debug print?
        self.DEBUG = False
        
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

        _service_stub = rf.Network(DEFAULT_SERVICE_STUB).interpolate(self.frequency)

        self.service_stub_left = _service_stub.copy()
        self.service_stub_left.name = 'service_stub_left'
        self.service_stub_right = _service_stub.copy()
        self.service_stub_right.name = 'service_stub_right'

        # service stub shorts
        self.short_left = rf.Circuit.Ground(self.frequency, name='short_left')
        self.short_right = rf.Circuit.Ground(self.frequency, name='short_right')

        # additional elements which will be usefull later
        self.port_left = rf.Circuit.Port(self.frequency, 'port_left', z0=self.windows_impedance_transformer_left.z0[:,0])
        self.port_right = rf.Circuit.Port(self.frequency, 'port_right', z0=self.windows_impedance_transformer_right.z0[:,0])

        # antenna front-face
        front_face = front_face or DEFAULT_FRONT_FACE
        if type(front_face) == str:
            # if a string, this should be a path to a Touchstone file
            self._antenna = rf.Network(front_face)
        elif type(front_face) == rf.network.Network:
            # if a Network
            self._antenna = front_face.copy()

        self._antenna.name = self._antenna.name or 'antenna'  # set a name if not exist

        # Renormalize to 50 ?
        # self.bridge_left.renormalize(50)
        # self.bridge_right.renormalize(50)
        # self.windows_impedance_transformer_left.renormalize(50)
        # self.windows_impedance_transformer_right.renormalize(50)
        # self._antenna.renormalize(50)

        # if the antenna front-face Network is defined on a single point (ex: from TOPICA)
        # duplicate this points to the other frequencies
        front_face_freq = self._antenna.frequency
        if len(front_face_freq) == 1:
            new_freq = self._frequency#rf.Frequency(front_face_freq.f, front_face_freq.f, unit='Hz', npoints=2)
            new_ntw = rf.Network(frequency=new_freq)
            new_ntw.z0 = np.repeat(self._antenna.z0, len(new_freq), axis=0)
            new_ntw.s = np.repeat(self._antenna.s, len(new_freq), axis=0)
            new_ntw.name = self._antenna.name or 'antenna'  # set a name if not exist
            self.antenna = new_ntw
        else:
            # interpolate the results for the frequency band
            self.antenna = self._antenna.interpolate(self._frequency)

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
        # service stub 3rd ports are left open
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
            # [(self.windows_impedance_transformer_left, 0), (self.port_left, 0)],  # no stub
            # [(self.windows_impedance_transformer_right, 0),  (self.port_right, 0)],  # no stub
            [(self.windows_impedance_transformer_left, 0), (self.service_stub_left, 1)],
            [(self.service_stub_left, 0), (self.port_left, 0)],
            [(self.windows_impedance_transformer_right, 0), (self.service_stub_right, 1)],
            [(self.service_stub_right, 0), (self.port_right, 0)],
            [(self.service_stub_left, 2), (self.short_left, 0)],
            [(self.service_stub_right, 2), (self.short_right, 0)],
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
        # print(C)
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

    
    def _optim_fun_both_sides(self, Cs, f_match=55e6, z_match=[29.89+0j, 29.89+0j], power=[1,1], phase=[0,np.pi]):
        '''
        Optimisation function to match both antenna sides.
        
        Optimization is made for active Z parameters, that is taking into account the antenna excitation.

        Parameters
        ----------
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF].
        f_match: float, optional
            match frequency in [Hz]. Default is 55 MHz.
        z_match: array of complex, optional
            antenna feeder characteristic impedance to match on. Default is [30,30] ohm 
        power : list or array
            Input power at external ports in Watts [W]. Default is [1,1] W.
        phase : list or array
            Input phase at external ports in radian [rad]. Default is dipole [0,pi] rad.

        Returns
        -------
        r: float
            Residuals of Z_act - Z_match
            r = (Z_act_left_re - np.real(z_match[0]))**2 + (Z_act_left_im - np.imag(z_match[0]))**2 +
                (Z_act_right_re - np.real(z_match[1]))**2 + (Z_act_right_im - np.imag(z_match[1]))**2
        '''
        # Create Antenna network for the capacitances Cs
        #z_act = self.z_act(power, phase, Cs=list(Cs))
        s_act = self.s_act(power, phase, Cs=list(Cs))
        
        # retrieve Z and compare to objective
        index_f_match = np.argmin(np.abs(self.f - f_match))

        #Z_act_left_re = np.real(z_act[index_f_match,0])
        #Z_act_left_im = np.imag(z_act[index_f_match,0])
        #Z_act_right_re = np.real(z_act[index_f_match,1])
        #Z_act_right_im = np.imag(z_act[index_f_match,1])
        
        #r = ((Z_act_left_re - np.real(z_match[0])) + (Z_act_left_im - np.imag(z_match[0])))**2 \
        #    * ((Z_act_right_re - np.real(z_match[1])) + (Z_act_right_im - np.imag(z_match[1])))**2
        
        r = np.sqrt(np.sum(np.abs(s_act[index_f_match,:])**2))
        
        if self.DEBUG:
            print(Cs, r)
        return r

    def match_one_side(self, f_match=55e6, solution_number=1, side='left',
                       z_match=29.89+0j, decimals=None):
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
        decimals : int, optional
            Round the capacitances to the given number of decimals. Default is None (no rounding)

        Returns
        -------
        Cs_match : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF].

        '''
        # creates an antenna circuit for single frequency only to speed-up calculations
        freq_match = rf.Frequency(f_match, f_match, npoints=1, unit='Hz')
        self._antenna_match = WestIcrhAntenna(freq_match, front_face=self.antenna)

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

            sol = scipy.optimize.minimize(self._optim_fun_one_side, C0,
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

        # round result to realistic values if requested
        if decimals:
            Cs = list(np.round(Cs, decimals=decimals))
            print('Rounded result:', Cs)

        return Cs

    def match_both_sides_separately(self, f_match=55e6, solution_number=1,
                                    z_match=[29.89+0j, 29.87+0j], decimals=None):
        """
        Match both sides separatly and returns capacitance values for each sides.

        Match the left side with right side unmatched, then match the right side
        with the left side unmatched. Combine the results

        Parameters
        ----------
        f_match: float, optional
            match frequency in [Hz]. Default is 55 MHz.
        solution_number: int, optional
            1 or 2: 1 for C_top > C_lower or 2 for C_top < C_lower
        z_match: complex, optional
            antenna feeder characteristic impedance to match on. Default is 30 ohm
        decimals : int, optional
            Round the capacitances to the given number of decimals. Default is None (no rounding)


        Returns
        -------
        Cs_match : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF].

        """
        C_left = self.match_one_side(side='left', f_match=f_match, solution_number=solution_number, z_match=z_match[0], decimals=decimals)
        C_right = self.match_one_side(side='right', f_match=f_match, solution_number=solution_number, z_match=z_match[1], decimals=decimals)

        C_match = [C_left[0], C_left[1], C_right[2], C_right[3]]
        return C_match

    def match_both_sides(self, f_match=55e6, solution_number=1, z_match=[29.89+0j, 29.89+0j], decimals=None,
                        power=[1,1], phase=[0, np.pi]):
        '''
        Match both sides at the same time for a given frequency target.
        
        Optimization is made for active Z parameters, that is taking into account the antenna excitation.
                
        Parameters
        ----------
        f_match: float, optional
            match frequency in [Hz]. Default is 55 MHz.
        solution_number: int, optional
            1 or 2: 1 for C_top > C_lower or 2 for C_top < C_lower
        z_match: array of complex, optional
            antenna feeder characteristic impedance to match on. Default is 30 ohm
        decimals : int, optional
            Round the capacitances to the given number of decimals. Default is None (no rounding)
        power : list or array
            Input power at external ports in Watts [W]. Default is [1, 1] W.
        phase : list or array
            Input phase at external ports in radian [rad]. Defalt is dipole [0, pi] rad.

        Returns
        -------
        Cs_match : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF].

        '''
        # creates an antenna circuit for a single frequency only to speed-up calculations
        freq_match = rf.Frequency(f_match, f_match, npoints=1, unit='Hz')
        self._antenna_match = WestIcrhAntenna(freq_match, front_face=self.antenna)

        # setup constraint optimization to force finding the requested solution
        sol_sign = +1 if solution_number == 1 else -1
        A = np.array([[1, 0, 0, 0], 
                      [0, 1, 0, 0], 
                      [0, 0, 1, 0], 
                      [0, 0, 0, 1], 
                      [sol_sign*-1, sol_sign*1, 0, 0], 
                      [0, 0, sol_sign*-1, sol_sign*1],
                     ])
        lb = np.array([12, 12, 12, 12, -np.inf, -np.inf])
        ub = np.array([150, 150, 150, 150, 0, 0])
        const = scipy.optimize.LinearConstraint(A, lb, ub)
        
        
        # initial guess from both side separately
        print('Looking for individual solutions separately for 1st guess...')
        C0 = self.match_both_sides_separately(f_match=f_match, solution_number=solution_number, 
                                              z_match=z_match, decimals=decimals)

        print('Searching for the active match point solution...')
        success = False
        while success == False:
            #sol = scipy.optimize.minimize(self._optim_fun_both_sides, C0,
            #                              args=(f_match, z_match, power, phase),
            #                              constraints=const, method='SLSQP',
            #                              options={'disp':True}
            #                              )
            sol = scipy.optimize.minimize(self._optim_fun_both_sides, C0,
                                          args=(f_match, z_match, power, phase),
                                          constraints=const, method='COBYLA',  
                                          options={'disp':True}
                                         )
            # test if the solution found is the capacitor range
            success = sol.success
            if np.isclose(sol.x, 150).any() or \
                np.isclose(sol.x, 12).any() or \
                np.isclose(sol.x[0], sol.x[1]) or \
                np.isclose(sol.x[2], sol.x[3]):
                success = False
                print('Wrong solution (out of range capacitor) ! Re-doing...')

            print(success, f'solution #{solution_number}:', sol.x)

        Cs = [sol.x[0], sol.x[1], sol.x[2], sol.x[3]]
        
        # round result to realistic values if requested
        if decimals:
            Cs = list(np.round(Cs, decimals=decimals))
            print('Rounded result:', Cs)
    
        return Cs

    def optimum_frequency_index(self, power, phase, Cs=None):
        """
        Array indexes of the optimum frequency with respect to active S-parameters for a given excitation

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
        f_opt_idx : array (2x1)
            array indexes of the optimum frequencies for each sides of the antenna

        """
        # use internal capacitances if not passed
        Cs = Cs or self.Cs
        # active S-parameters for dipole excitation
        s_act = self.s_act(power, phase, Cs=Cs)

        f_opt_idx = np.argmin(np.abs(s_act), axis=0)
        return f_opt_idx

    def optimum_frequency(self, power, phase, Cs=None):
        """
        Optimum frequency with respect to active S-parameters for a given excitation

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
        f_opt : array (2x1)
            optimum frequencies for each sides of the antenna

        """
        # use internal capacitances if not passed
        Cs = Cs or self.Cs

        f_opt = self.frequency.f[self.optimum_frequency_index(power, phase, Cs=Cs)]
        return f_opt


    def _Xs(self, f=None):
        """
        Strap Reactance fit with frequency

        Parameters
        ----------
        f : array, opt
            frequency in Hz. Default is None (use internal frequency)

        Returns
        -------
        Xs : real array
            strap reactance (nb_f,1)

        """
        f = f or self.frequency.f
        # scaled frequency
        f_MHz = f/1e6
        Xs = 1.66e-04*f_MHz**3 -1.53e-02*f_MHz**2 + 1.04*f_MHz -7.77
        return Xs

    def load(self, Rc, Xs=None):
        '''
        Load the antenna model with an ideal plasma load (no poloidal and toroidal cross coupling)

        Parameters
        ----------
        Rc : float
            Coupling Resistance [Ohm]
        Xs : float, optional
            Strap reactance. The default is None (use frequency best fit).

        Returns
        -------
        None.

        '''
        # reactance : if not passed, use best fit
        Xs = Xs or self._Xs()
        # interpolating the default z0 with the one of the CAD model
        # to keep the same z0 behaviour
        f = rf.interp1d(self._antenna.f, self._antenna.z0[:,0])
        _z0 = f(self.f)
        # port and short definitions
        _port1 = rf.Circuit.Port(self.frequency, 'Port1', z0=_z0)
        _port2 = rf.Circuit.Port(self.frequency, 'Port2', z0=_z0)
        _port3 = rf.Circuit.Port(self.frequency, 'Port3', z0=_z0)
        _port4 = rf.Circuit.Port(self.frequency, 'Port4', z0=_z0)
        _short1 = rf.Circuit.Ground(self.frequency, 'Gnd1', z0=_z0)
        _short2 = rf.Circuit.Ground(self.frequency, 'Gnd2', z0=_z0)
        _short3 = rf.Circuit.Ground(self.frequency, 'Gnd3', z0=_z0)
        _short4 = rf.Circuit.Ground(self.frequency, 'Gnd4', z0=_z0)
        # load definition
        z_s = Rc + 1j*Xs
        media = rf.DefinedGammaZ0(frequency=self.frequency, z0=_z0)
        _load1 = media.resistor(z_s, name='load1', z0=_z0)
        _load2 = media.resistor(z_s, name='load2', z0=_z0)
        _load3 = media.resistor(z_s, name='load3', z0=_z0)
        _load4 = media.resistor(z_s, name='load4', z0=_z0)

        cnx = [
                [(_port1, 0), (_load1, 0)], [(_load1, 1), (_short1, 0)],
                [(_port2, 0), (_load2, 0)], [(_load2, 1), (_short2, 0)],
                [(_port3, 0), (_load3, 0)], [(_load3, 1), (_short3, 0)],
                [(_port4, 0), (_load4, 0)], [(_load4, 1), (_short4, 0)],
            ]

        crt = rf.Circuit(cnx)
        _antenna = crt.network
        _antenna.name = 'antenna'
        self.antenna = _antenna

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
        Is : complex array (nb_f,4)
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
        Vs : complex array (nb_f,4)
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

    @property
    def f_scaled(self):
        '''
        Antenna Frequency band scaled to the Frequency unit

        Returns
        -------
        f_scaled : array (nb_f,)
            Antenna frequency band values in the Frequency unit

        '''
        return self._frequency.f_scaled


    def s_act(self, power, phase, Cs=None):
        '''
        Active S-parameters for a given excitation

        Active s-parameters are defined by :

        .. math::

            \mathrm{active}(s)_{mn} = \sum_i s_{mi} \\frac{a_i}{a_n}


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
        s_act : complex array (nb_f, 2)
            Active S-parameters
        '''
        Cs = Cs or self.Cs
        a = self.circuit(Cs)._a_external(power, phase)
        return self.circuit(Cs).s_active(a)

    def z_act(self, power, phase, Cs=None):
        '''
        Active Z-parameters for a given excitation

        The active Z-parameters are defined by:

        .. math::

            \mathrm{active}(z)_{m} = z_{0,m} \\frac{1 + \mathrm{active}(s)_m}{1 - \mathrm{active}(s)_m}

        where :math:`z_{0,m}` is the characteristic impedance and
        :math:`\mathrm{active}(s)_m` the active S-parameter of port :math:`m`.

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
        z_act : complex array (nb_f, 2)
            Active Z-parameters
        '''
        Cs = Cs or self.Cs
        a = self.circuit(Cs)._a_external(power, phase)
        return self.circuit(Cs).z_active(a)

    def vswr_act(self, power, phase, Cs=None):
        """
        Active VSWR for a given excitation

        The active VSWR is defined by :

        .. math::

            \mathrm{active}(vswr)_{m} = \\frac{1 + |\mathrm{active}(s)_m|}{1 - |\mathrm{active}(s)_m|}

        where :math:`\mathrm{active}(s)_m` the active S-parameter of port :math:`m`.

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
        vswr_act : complex array (nb_f,2)
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
        Vs : complex array (nb_f,4)
            Voltages at antenna front face ports [V1, V2, V3, V4]

        Example
        -------
        >>> Vs = west_antenna.voltages([1, 1], [0, pi])

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
        Is : complex array (nb_f,4)
            Currents at antenna front face ports [I1, I2, I3, I4]

        Example
        -------
        >>> Is = west_antenna.currents([1, 1], [0, pi])

        """
        _Cs = Cs or self.Cs
        idx_antenna = [0, 4, 2, 6]  # for port 1,2,3,4 of the antenna
        return self.circuit(_Cs).currents(power, phase)[:, idx_antenna]

    def Z_T(self, power, phase, Cs=None):
        """
        Impedances Z_T at the T-junction

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
        Z_T : complex array (nb_f,2)
            Impedance at the T-junction [Z_T_left, Z_T_right]

        Example
        -------
        >>> Z_T = west_antenna.Z_T([1, 1], [0, pi])

        """
        _Cs = Cs or self.Cs
        Is = self.circuit(Cs=_Cs).currents(power, phase)
        Vs = self.circuit(Cs=_Cs).voltages(power, phase)
        Zs = Vs/Is
        # indexes 7 and 8 corresponds to the connections between impedance
        # transformer and bridges for left and right sides respectively
        Z_T = Zs[:,(7,8)]
        return Z_T

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

        where
        - Pt is the transmitted (coupled) power
        - Is is the current average Is^2 = |I_top|^2 + |I_bot|^2

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
        Is_left = np.sqrt(np.abs(Is[:,0])**2 + np.abs(Is[:,1])**2)
        Is_right =  np.sqrt(np.abs(Is[:,2])**2 + np.abs(Is[:,3])**2)

        # coupled power
        Pr = self.Pr(power, phase, Cs=_Cs)
        Pi = power
        Pt = Pi - Pr
        # coupling resistance
        Rc_left = 2*Pt[:,0]/Is_left**2
        Rc_right = 2*Pt[:,1]/Is_right**2

        return np.c_[Rc_left, Rc_right]

    def Rc_WEST(self, power, phase, Cs=None):
        """
        Coupling Resistances of both sides of the antenna - WEST Approximation

        Rc = 2Pt/Is^2 where Pt is the transmitted power and Is the average current

        where
        - Pt is the transmitted (coupled) power
        - Is is the current average Is^2 = |I_top|^2 + |I_bot|^2

        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]. Default is None (use internal Cs)
                    [C1] [C3]
                    [C2] {C4} (view from the rear of the antenna)

        Returns
        -------
        Rc: array (nb_f, 2)
            Coupling resistances [Ohm] of both sides vs freq

        """
        _Cs = Cs or self.Cs
        # average currents
        Vs = self.voltages(power, phase, Cs=_Cs)
        Vs_left = np.sqrt(np.abs(Vs[:,0])**2 + np.abs(Vs[:,1])**2)
        Vs_right =  np.sqrt(np.abs(Vs[:,2])**2 + np.abs(Vs[:,3])**2)
        # Reactance
        Xs = self._Xs()

        # coupled power
        Pr = self.Pr(power, phase, Cs=_Cs)
        Pi = power
        Pt = Pi - Pr
        # coupling resistance
        Rc_left = 2 * Xs**2 * Pt[:,0]/Vs_left**2
        Rc_right = 2 * Xs**2 * Pt[:,1]/Vs_right**2

        return np.c_[Rc_left, Rc_right]

    def front_face_Rc(self, Is=[+1,-1,-1,+1]):
        """
        (Ideal) front-face coupling resistances

        Coupling resistance is defined as :

            Rc = 2*Pt / Is^2

        where
        - Pt is the transmitted (coupled) power
        - Is is the current average Is^2 = |I_top|^2 + |I_bot|^2

        Warning
        -------
        Pay attention to the port indexing!
        `Is` is assumed with the following order:
            [1] [2]
            [3] {4}
        which is the one sued in HFSS or in TOPICA models,

        However, the indexing order of voltages probes and capacitors is:
            [1] [3]
            [2] {4}
        (both view from the rear of the antenna)

        Parameters
        ----------
        Is : list or array (complex)
            Current excitation at antenna front-face ports.
            Default is [+1,-1,-1,+1] (dipole)

        """

        V = self.antenna.z @ Is
        Prf = 0.5 * V @ Is

        sum_I_left_avg_square = np.abs(Is[0])**2 + np.abs(Is[2])**2
        sum_I_right_avg_square = np.abs(Is[1])**2 + np.abs(Is[3])**2

        Rc_left = 2*Prf.real/sum_I_left_avg_square
        Rc_right = 2*Prf.real/sum_I_right_avg_square
        return np.c_[Rc_left, Rc_right]


    def swit_abcd(self):
        """
        ABCD matrix of service Stub, Window and Impedance Transformer (aka "SWIT")

        Returns
        -------
        a : np.array (nb_f,2,2)
            ABCD matrix of the SWIT

        """
        return self.windows_impedance_transformer.a
    
    def z_coupler(self, power, phase, Cs=None):
        """
        Input impedance at the bidirective coupler for a given excitation. 

        Assume that the bidirective coupler is located just behind the RF window.
        In reality, there is a piece of transmission line from the window to 
        the couplern which depends of each sides of each antenna in WEST.

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
        z_coupler : np.array (nb_f, 2)
            Input impedance (left and right side) at the bidirective coupler for a given excitation. 

        """
        # TODO (?): include the piece of transmission line for each antenna? 
        return self.z_act(power, phase, Cs=Cs)
    
    def z_T(self, power, phase, Cs=None):
        """
        Input impedance at the T-junction (input of the bridge) for a given excitation.

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
        z_T : np.array (nb_f, 2)
            Input impedances (left and right sides) at the T-junction for a given excitation.

        """
        # ABCD matrix of service Stub, Window and Impedance Transformer (aka "SWIT")
        A = self.swit_abcd()  # shape (nb_f,2,2)
        z_coupler = self.z_coupler(power, phase, Cs)  # shape (nb_f,2)
        
        # From ABCD matrix definition for SWIT:
        # (V_coupler, I_coupler) = [A,B; C,D]*(V_T, -I_T)
        # --> Z_T = (A12 -  A22 Z_coupler)/(A21 Z_coupler - A11)
        z_T_left = (A[:,0,1] - A[:,1,1]*z_coupler[:,0])/(A[:,1,0]*z_coupler[:,0] - A[:,0,0])
        z_T_right = (A[:,0,1] - A[:,1,1]*z_coupler[:,1])/(A[:,1,0]*z_coupler[:,1] - A[:,0,0])
        return np.c_[z_T_left, z_T_right]

    def error_signals(self, power, phase, Cs=None, z_T_target=Z_T_OPT):
        """
        Normalized Error Signals for left and right sides.
        
        Normalized Error signals is defined by:
        
        epsilon = (z_T_target - z_T)/z_T

        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]. Default is None (use internal Cs)
        z_T_target : complex, optional
            Desired target (Set Point) for the input impedance at T-junction. The default is 2.89-0.17j.

        Returns
        -------
        epsilons : complex array (nb_f, 2)
            Error signals for left and right sides
        
        """
        z_T = self.z_T(power, phase, Cs)
        return (z_T_target - z_T)/z_T
    
    def _T(self, k=5, S=-1, alpha=45, Kri=1):
        """
        Feedback Loop Matrix T
        
        C_match = C + T * epsilon 

        Parameters
        ----------
        k : float, optional
            Gain. The default is 5.
        S : int, optional
            Solution choice. The default is -1 for solution 1, +1 for solution 2.
        alpha : float, optional
            Solution choice. The default is 45 for solution 1, -45 for solution 2.
        Kri : float, optional
            Ratio of real to imag relative weight. The default is 1.

        Returns
        -------
        T : array (2,2)
            Feedback Loop Matrix.

        """       
        T = np.array([[-S*np.cos(np.deg2rad(alpha)), S*Kri*np.sin(np.deg2rad(alpha))],
                      [-np.sin(np.deg2rad(alpha)), -Kri*np.cos(np.deg2rad(alpha))]])
        return T
    
    def capacitor_predictor(self, power, phase, Cs, z_T_target=Z_T_OPT, solution_number=1, K=1):
        """
        Return a capacitance set toward matching.
        
        The predictor does not return the final optimum values but only 
        a set which goes toward the solution. Hence, this function should
        be re-itered until its converges (if so).

        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF].
        z_T_target : complex, optional
            Desired target (Set Point) for the input impedance at T-junction. The default is 2.89-0.17j.
        solution_number : int
            Desired solution. 1 for C_top > C_bot and 2 for the opposite. 
        K : float
            Gain

        Returns
        -------
        C_left : np.array (nb_f, 2)
            Left side capacitances  (top, bottom)
        C_right : np.array (nb_f, 2)
            Right side capacitances  (top, bottom)

        """
        C_left_current = np.array(Cs[:2])
        C_right_current = np.array(Cs[2:])
        
        # D is the sign of the elements of the Jacobian calculated
        # for C - C_SP  (from Taylor theorem)
        if solution_number == 1:
            D = K*np.array([[-1, -1],
                            [-1, +1]])
        elif solution_number == 2:
            D = K*np.array([[-1, -1],
                            [+1, -1]])

        # relative error of Z_T vs set point Z_T_SP (desired value)
        epsilons = self.error_signals(power, phase, Cs, z_T_target)
        
        # (2,2)@(2,nb_f)=(2,nb_f)
        # Note a sign difference in front of D, since here it's not C - C_SP
        # but the error signals which is (Z_T_SP - ZT)/Z_T
        C_left = C_left_current + (D @ np.c_[np.real(epsilons[:,0]), np.imag(epsilons[:,0])].T).T
        C_right = C_right_current + (D @ np.c_[np.real(epsilons[:,1]), np.imag(epsilons[:,1])].T).T
        
        return C_left, C_right        
    
    def capacitor_velocities(self, power, phase, Cs=None, z_T_target=Z_T_OPT, K=1):
        """
        Velocity requests toward matching point.   

        Parameters
        ----------
        power : list or array
            Input power at external ports in Watts [W]
        phase : list or array
            Input phase at external ports in radian [rad]
        Cs : list or array
            antenna 4 capacitances [C1, C2, C3, C4] in [pF]. Default is None (use internal Cs)
        z_T_target : complex, optional
            Desired target (Set Point) for the input impedance at T-junction. The default is 2.89-0.17j.
        K : float, optional
            Gain. Default is 1.

        Returns
        -------
        v_left: np.array (nb_f, 2)
            Left side velocities (v_top, v_bot)
        v_right: np.array (nb_f, 2)
            Right side velocities (v_top, v_bot)            

        """
        T = self._T()
        epsilons = self.error_signals(power,phase,Cs,z_T_target)
        # (2,2)@(2,nb_f)=(2,nb_f)
        v_left = K * T @ np.c_[np.real(epsilons[:,0]), np.imag(epsilons[:,0])].T
        v_right = K * T @ np.c_[np.real(epsilons[:,1]), np.imag(epsilons[:,1])].T
        
        return v_left.T, v_right.T  # (nb_f, 2)