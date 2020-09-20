# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 21:00:22 2013

@author: hash
"""
import numpy as np
import skrf as rf


class TopicaResult:

    def __init__(self, filename, z0=50):
        """
        TOPICA result constructor.
        
        Parameters
        ----------
        filename (str): 
            filename of the ascii file
        [z0 (float)]: 
            characteristic impedance of the port in Ohm (default: 50)

        """
        self.filename = filename
        self.nbPorts = len(self.z)
        self._z0 = z0

    @property
    def z0(self):
        """
        Characteristic Impedance of the TOPICA Model's ports

        Returns
        -------
        z0 : array

        """
        return self._z0

    @z0.setter
    def z0(self, z0):
        """ 
        Set the characteristic impedance of the ports in Ohm.

        Parameters
        ----------        
        z0 : :class:`numpy.ndarray` of length n
                characteristic impedance for network
                
        """
        self._z0 = z0


    @property
    def raw_data(self):
        """
        raw TOPICA data from txt file

        Returns
        -------
        data : array
            raw TOPICA data : [index_row, index_col, real_z, z_imag]

        """
        return np.loadtxt(self.filename, skiprows=3)

    @property
    def z(self):
        """ 
        Get the impedance matrix as an NxN array, N being the number of ports.
        
        Returns
        ----------        
        z : :class:`numpy.ndarray` 
                impedance matrix
                
        """        
        data = self.raw_data
        # fill the Z matrix (Nport x Nport) 
        z = np.zeros((int(np.sqrt(data.shape[0])), int(np.sqrt(data.shape[0]))), dtype='complex')
        for id1 in range(data.shape[0]):
            z[int(data[id1, 0])-1, int(data[id1, 1])-1] = complex(data[id1, 2], data[id1, 3])
        return(z)
    
    @property
    def s(self):
        """ 
        Get the scattering parameters as an NxN array, N being the number of ports.
        
        Returns
        ---------
        s : :class:`numpy.ndarray` of shape n x n
                scattering parameters of the TOPICA result
                
        """
        Zref = np.diag(np.repeat(self.z0, self.nbPorts))

        G = 1/np.sqrt(np.real(self.z0)) 
        Gref = np.diag(np.repeat(G, self.nbPorts))
        # Z to S formulae
        S = Gref @ (self.z - Zref) @ (np.linalg.inv(self.z + Zref)) @ (np.linalg.inv(Gref))

        return(S)

    def to_skrf_network(self, skrf_frequency, name='front-face'):
        """ 
        Convert into a skrf Network.
        
        Assume the scattering parameters of the network are the same 
        for all the frequencies of the network. 
        
        Parameters
        ----------
        skrf_frequency : :class: 'skrf.frequency' 
            Frequency of the network (for compatibility with skrf)
        name : string
            name of the network
        
        Returns
        ----------
        network : :class: 'skrf.network' 
        
        """
        network = rf.Network(name=name)
        network.s = np.tile(self.s, (len(skrf_frequency), 1, 1))
        network.z0 = self.z0
        network.frequency = skrf_frequency
        return(network)

    def write_touchstone(self, filename, skrf_frequency, name=''):
        """
        Write the scattering parameters into a Touchstone file format
        (.sNp where N is the number of ports).

        Parameters
        ----------
        filename : string
            Touchstone filename
        skrf_frequency : :class: ' skrf.frequency' 
            skrf Frequency object
        name : string: 
            name of the network
        """
        network = self.to_skrf_network(skrf_frequency, name=name)
        network.write_touchstone(filename=filename, write_z0=True)
        
    def Rc(self, I=[1, -1, -1, 1]):
        """
        Calculate the coupling resistance of the TOPICA Z-matrix for 
        a given current excitation.
        
        Coupling resistance is defined as in :
        W. Helou, L. Colas, J. Hillairet et al., Fusion Eng. Des. 96–97 (2015) 5–8. doi:10.1016/j.fusengdes.2015.01.005.
        
        Parameters
        ----------
        I : list
            Current excitation at port. Default is [1, -1, -1, 1], ie. dipole phasing of 2x2 straps antenna
        
        Returns
        -------
        Rc : float
            Coupling Resistance
        """
        N = len(I) # number of ports
        V = self.z @ I
        Pt = np.real(V.conj() @ I) / 2
        Is = np.sqrt(np.sum(np.abs(I)) / N)
        # divided by 2 as the current average is made on 4 straps (2 per side)
        Rc = Pt / (2*Is**2)
        return Rc
        
