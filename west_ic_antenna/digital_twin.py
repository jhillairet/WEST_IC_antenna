# -*- coding: utf-8 -*-
"""
WEST ICRH Antenna RF Model Digital Twin

@author: Julien Hillairet

This class shall be used in the frame of a jupyter notebook
in order to get the interactive functionalities

.. module:: west_ic_antenna.digital_twin

.. autosummary::
    :toctree:
    
    DigitalTwin

"""
import ipywidgets as widgets
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from . antenna import WestIcrhAntenna
from IPython.core.debugger import set_trace

class DigitalTwin(widgets.HBox):

    def __init__(self):
        super().__init__()
        output = widgets.Output()

        #### defining widgets
        # capacitor widgets
        C1 = widgets.FloatSlider(value=49.31, min=30.0, max=120.0, step=1e-4,
                                 continuous_update=False)
        C2 = widgets.FloatSlider(value=47.38, min=30.0, max=120.0, step=1e-4,
                                 continuous_update=False)
        C3 = widgets.FloatSlider(value=49.11, min=30.0, max=120.0, step=1e-4,
                                 continuous_update=False)
        C4 = widgets.FloatSlider(value=47.56, min=30.0, max=120.0, step=1e-4,
                                 continuous_update=False)

        def capa_plus(clicked_button): 
            eval(clicked_button.name, 
                {'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4}).value += 0.1
                
        def capa_minus(clicked_button):
            eval(clicked_button.name, 
                {'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4}).value -= 0.1
            
        def plus_minus_button(name, sign):
            '''
            Generic Button Constructor

            Parameters
            ----------
            name : str
                Button name 
            sign : str
                'plus' of 'minus'

            Returns
            -------
            button : ipywidgets.widgets.Button
                + or - Button

            '''
            if sign == 'plus':
                tooltip = '+ 0.1 pF'
                icon = 'plus'
            elif sign == 'minus':
                tooltip = '- 0.1 pF'
                icon = 'minus'                
            
            button = widgets.Button(
                description='', disabled=False,
                button_style='', tooltip=tooltip,
                icon=icon, layout=widgets.Layout(width='30px')
            )
            button.name = name
            
            if sign == 'plus':
                button.on_click(capa_plus)
            elif sign == 'minus':
                button.on_click(capa_minus)
            
            return button
            
        C1_minus = plus_minus_button('C1', 'minus')
        C1_plus = plus_minus_button('C1', 'plus')
        C2_minus = plus_minus_button('C2', 'minus')
        C2_plus = plus_minus_button('C2', 'plus')
        C3_minus = plus_minus_button('C3', 'minus')
        C3_plus = plus_minus_button('C3', 'plus')
        C4_minus = plus_minus_button('C4', 'minus')
        C4_plus = plus_minus_button('C4', 'plus')
        button_plot = widgets.Button(description='plot')
        
        capas = widgets.HBox([widgets.VBox([widgets.Box([widgets.Label('C1: '), C1, C1_minus, C1_plus]),
                                            widgets.Box([widgets.Label('C2: '), C2, C2_minus, C2_plus])]),
                              widgets.VBox([widgets.Box([widgets.Label('C3: '), C3, C3_minus, C3_plus]),
                                            widgets.Box([widgets.Label('C4: '), C4, C4_minus, C4_plus])]),
                              button_plot
                             ])

        # excitation widgets
        power_left = widgets.FloatSlider(description='Left Left Fwd Power [kW]:', value=10, min=0, max=1500, step=1, continuous_update=False)
        power_right = widgets.FloatSlider(description='Right Side Fwd Power [kW]:', value=10, min=0, max=1500, step=1, continuous_update=False)  
        phase_rel = widgets.FloatSlider(value=180, min=0, max=360, step=1,
                                        continuous_update=False)
        excitation = widgets.VBox([
            widgets.HBox([power_left, power_right]),
            widgets.Box([widgets.Label('Phase: '), phase_rel])
            ])

        # matching widgets
        match_freq = widgets.FloatSlider(min=40, max=65, value=55.5, step=1e-1,
                                        continuous_update=False)
        match_type = widgets.RadioButtons(options=['Left Side only', 'Right Side only', 
                                                   'Both Sides Separately', 
                                                   'Both Sides Simultaneously', 
                                                   'Both Sides Simultaneously (iterative)'])
        match_sol = widgets.RadioButtons(options=['1', '2'])
        match_button = widgets.Button(description='Match', disabled=False,
                               button_style='', tooltip='Click me', icon='check')

        # frequency widgets
        front_face_vacuum = '../west_ic_antenna/data/Sparameters/front_faces/WEST_ICRH_antenna_front_face_curved_30to70MHz.s4p'
        front_face_aquarium ='../west_ic_antenna/data/Sparameters/front_faces/aquarium/HFSS/Epsr_55MHz/WEST_ICRH_front_face_with_aquarium_Daq00cm.s4p'
        front_face_plasma = '../west_ic_antenna/data/Sparameters/front_faces/TOPICA/S_TSproto12_55MHz_Hmode_LAD6.s4p'
        front_face_plasma_Lmode1 = '../west_ic_antenna/data/Sparameters/front_faces/TOPICA/S_TSproto12_55MHz_Profile1.s4p'
        front_face_plasma_Lmode2 = '../west_ic_antenna/data/Sparameters/front_faces/TOPICA/S_TSproto12_55MHz_Profile2.s4p'
        front_face_plasma_Lmode3 = '../west_ic_antenna/data/Sparameters/front_faces/TOPICA/S_TSproto12_55MHz_Profile3.s4p'
        front_face_plasma_Lmode4 = '../west_ic_antenna/data/Sparameters/front_faces/TOPICA/S_TSproto12_55MHz_Profile4.s4p'
        front_face_plasma_Lmode5 = '../west_ic_antenna/data/Sparameters/front_faces/TOPICA/S_TSproto12_55MHz_Profile5.s4p'
        front_face_plasma_Lmode6 = '../west_ic_antenna/data/Sparameters/front_faces/TOPICA/S_TSproto12_55MHz_Profile6.s4p'
        front_face_plasma_Lmode7 = '../west_ic_antenna/data/Sparameters/front_faces/TOPICA/S_TSproto12_55MHz_Profile7.s4p'
        front_face_plasma_Lmode8 = '../west_ic_antenna/data/Sparameters/front_faces/TOPICA/S_TSproto12_55MHz_Profile8.s4p'

        front_face = widgets.Dropdown(
            options=[('Vacuum', front_face_vacuum),
                     ('Aquarium', front_face_aquarium),
                     ('Plasma', front_face_plasma),
                     ('L-mode 1', front_face_plasma_Lmode1),
                     ('L-mode 2', front_face_plasma_Lmode2),
                     ('L-mode 3', front_face_plasma_Lmode3),
                     ('L-mode 4', front_face_plasma_Lmode4),
                     ('L-mode 5', front_face_plasma_Lmode5),
                     ('L-mode 6', front_face_plasma_Lmode6),
                     ('L-mode 7', front_face_plasma_Lmode7),
                     ('L-mode 8', front_face_plasma_Lmode8),
                     ],

            value=front_face_vacuum,
            description='Front Face:',
            disabled=False,
        )

        out_matching = widgets.Output()

        matching = widgets.VBox([
            widgets.Box([widgets.Label('Front Face:'), front_face]),
            widgets.Box([widgets.Label('Match frequency:'), match_freq]),
            widgets.Box([widgets.Label('Solution type:'), match_sol]),
            match_type,
            match_button,
            out_matching
        ])


        frequency_range = widgets.FloatRangeSlider(
            value=[match_freq.value-1, match_freq.value+1],
            min=30.0,
            max=70.0,
            step=0.1,
            description='',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        frequency_npoints = widgets.IntText(value=101)

        frequency = widgets.VBox([
            widgets.Box([widgets.Label('Frequency range:'), frequency_range]),
            widgets.Box([widgets.Label('Number of points:'), frequency_npoints])
        ])

        # Init the figure
        fig, axes = plt.subplots(6, 1, sharex=True, figsize=(8,6));
        [a.set_ylim(-30, 2) for a in axes[:1]]
        [a.grid(True) for a in axes]
        axes[0].set_ylabel('$S_{ii}$ [dB]')
        axes[1].set_ylabel('$S_{act,i}$ [dB]')
        axes[2].set_ylabel('Voltage [kV]')
        axes[3].set_ylabel('Currents [A]')
        axes[4].set_ylabel('Phase [deg]')
        axes[5].set_ylabel('Err Sign')
        #axes[5].set_ylabel('Phase [deg]')
        #axes[5].set_ylim(-180, 180)
        #axes[6].set_ylabel('Rc [Ohm]')
        axes[-1].set_xlabel('Frequency [MHz]')
        fig.subplots_adjust(hspace=0)

        antenna = WestIcrhAntenna()

        def init_antenna(_):
            # Create antenna object for the given frequency span and number of points
            freq = rf.Frequency(start=frequency_range.value[0],
                                stop=frequency_range.value[1],
                                npoints=frequency_npoints.value,
                                unit='MHz')
            Cs = [C1.value, C2.value, C3.value, C4.value]

            return WestIcrhAntenna(frequency=freq, front_face=front_face.value, Cs=Cs)

        def plot_s(change):
            antenna = init_antenna([])
            antenna.Cs = [C1.value, C2.value, C3.value, C4.value]
            power = [power_left.value*1e3, power_right.value*1e3]
            phase = [0, np.deg2rad(phase_rel.value)]

            # Ignore Division per zero warning occuring in s_act
            with np.errstate(divide='ignore'):
                s_act = antenna.s_act(power, phase)
                s = antenna.circuit().s_external
                Vs = antenna.voltages(power, phase)
                Is = antenna.currents(power, phase)
                Rc = antenna.Rc_WEST(power, phase)
                Cs_left, Cs_right, _ = antenna.capacitor_predictor(power, phase, Cs=antenna.Cs)
                vleft, vright = antenna.capacitor_velocities(power, phase, Cs=antenna.Cs)

            """Remove old lines from plot and plot new ones"""
            [ax.clear() for ax in axes]
             
            # S11 and S22
            [[l.remove() for l in ax.lines] for ax in axes]
            [[l.remove() for l in ax.lines] for ax in axes]            
            [[l.remove() for l in ax.lines] for ax in axes]
            [[l.remove() for l in ax.lines] for ax in axes]
            
            axes[0].plot(antenna.f_scaled, 20*np.log10(np.abs(s[:,0,0])), color='C0')
            axes[0].plot(antenna.f_scaled, 20*np.log10(np.abs(s[:,1,1])), color='C1')
            # active S parameters
            axes[1].plot(antenna.f_scaled, 20*np.log10(np.abs(s_act[:,0])), color='C0')
            axes[1].plot(antenna.f_scaled, 20*np.log10(np.abs(s_act[:,1])), color='C1')
            # voltages and currents
            axes[2].plot(antenna.f_scaled, np.abs(Vs))
            axes[3].plot(antenna.f_scaled, np.abs(Is))  
            axes[4].plot(antenna.f_scaled, (np.angle(Vs[:,2], deg=True) - np.angle(Vs[:,0], deg=True))%360)
            axes[4].plot(antenna.f_scaled, (np.angle(Vs[:,3], deg=True) - np.angle(Vs[:,1], deg=True))%360)
            # axes[5].plot(antenna.f_scaled, (np.angle(Vs[:,1]/Vs[:,0], deg=True)))
            # axes[5].plot(antenna.f_scaled, (np.angle(Vs[:,2]/Vs[:,3], deg=True)))
            # axes[6].plot(antenna.f_scaled, Rc)

            axes[5].plot(antenna.f_scaled, vleft[:,0], color='C0')
            axes[5].plot(antenna.f_scaled, vleft[:,1], color='C1')
            axes[5].plot(antenna.f_scaled, vright[:,0], color='C2')
            axes[5].plot(antenna.f_scaled, vright[:,1], color='C3')                       
            
            #axes[5].plot(antenna.f_scaled, Cs_left[:,0] - antenna.Cs[0], color='C0', label='dC1')
            #axes[5].plot(antenna.f_scaled, Cs_left[:,1] - antenna.Cs[1], color='C1', label='dC2')
            #axes[5].plot(antenna.f_scaled, Cs_right[:,0] - antenna.Cs[2], color='C2', label='dC3')
            #axes[5].plot(antenna.f_scaled, Cs_right[:,1] - antenna.Cs[3], color='C3', label='dC4')           
            
            axes[2].legend(('V1', 'V2', 'V3', 'V4'), ncol=4)
            axes[3].legend(('I1', 'I2', 'I3', 'I4'), ncol=4)
            axes[4].legend(('∠(V3/V1)', '∠(V4/V2)'), ncol=2)
            # axes[5].legend(('∠(V2/V1)', '∠(V4/V3)'), ncol=2)
            # axes[6].legend(('Left', 'Right'), ncol=2)
            axes[5].set_ylim(-10, 10)
            [a.axvline(match_freq.value, ls='--', color='k') for a in axes]
        
            axes[0].set_xlim(left=frequency_range.value[0], right=frequency_range.value[1])

        # When one change the match frequency
        def cb_match_frequency(_):
            # update the frequency span of the figure
            frequency_range.value = [match_freq.value-1, match_freq.value+1]                
            
        # match the antenna
        def match(_):
            with out_matching:
                antenna = init_antenna([])
                Cs = [C1.value, C2.value, C3.value, C4.value]
                # what happens when we press the button
                clear_output()
                if match_type.value == 'Left Side only':
                    print(f'Searching for a match point left side at {match_freq.value*1e6}')
                    Cs_sol = antenna.match_one_side(f_match=match_freq.value*1e6,
                                                    side='left', solution_number=int(match_sol.value))
                    Cs[0] = Cs_sol[0]
                    Cs[1] = Cs_sol[1]
                    C1.value = Cs[0]
                    C2.value = Cs[1]
                elif match_type.value == 'Right Side only':
                    print(f'Searching for a match point right side at {match_freq.value*1e6}')
                    Cs_sol = antenna.match_one_side(f_match=match_freq.value*1e6,
                                                    side='right', solution_number=int(match_sol.value))
                    Cs[2] = Cs_sol[2]
                    Cs[3] = Cs_sol[3]
                    C3.value = Cs[2]
                    C4.value = Cs[3]
                elif match_type.value == 'Both Sides Separately':
                    print(f'Searching for a match point both sides separately at {match_freq.value*1e6}')
                    Cs_sol = antenna.match_both_sides_separately(f_match=match_freq.value*1e6,
                                                                 solution_number=int(match_sol.value))
                    Cs = Cs_sol
                    C1.value = Cs[0]
                    C2.value = Cs[1]
                    C3.value = Cs[2]
                    C4.value = Cs[3]
                elif match_type.value == 'Both Sides Simultaneously':
                    Cs_sol = antenna.match_both_sides(f_match=match_freq.value*1e6,
                                                      solution_number=int(match_sol.value), 
                                                      power=[power_left.value*1e3, power_right.value*1e3],
                                                      phase=[0, np.deg2rad(phase_rel.value)])
                    Cs = Cs_sol
                    C1.value = Cs[0]
                    C2.value = Cs[1]
                    C3.value = Cs[2]
                    C4.value = Cs[3]
                elif match_type.value == 'Both Sides Simultaneously (iterative)':
                    Cs_sol = antenna.match_both_sides_iterative(f_match=match_freq.value*1e6,
                                                                solution_number=int(match_sol.value), 
                                                                power=[power_left.value*1e3, power_right.value*1e3],
                                                                phase=[0, np.deg2rad(phase_rel.value)],
                                                                C0=Cs)
                    Cs = Cs_sol
                    C1.value = Cs[0]
                    C2.value = Cs[1]
                    C3.value = Cs[2]
                    C4.value = Cs[3]
                print(Cs)
                plot_s([])

        # Define callbacks
        match_freq.observe(cb_match_frequency)
        match_button.on_click(match)
        C1.observe(plot_s)
        C2.observe(plot_s)
        C3.observe(plot_s)
        C4.observe(plot_s)
        power_left.observe(plot_s)
        power_right.observe(plot_s)
        phase_rel.observe(plot_s)
        frequency_range.observe(plot_s)
        frequency_npoints.observe(plot_s)
        front_face.observe(plot_s)
        button_plot.on_click(plot_s)

        # setting the tab windows
        tab = widgets.Tab()
        tab.children = [capas, excitation, matching, frequency]
        tab.set_title(0, 'Capacitors')
        tab.set_title(1, 'Excitation')
        tab.set_title(2, 'Matching')
        tab.set_title(3, 'Frequency & front face')

        plot_s([])  # show something at the startup
        # tab  # must be last
        self.children = [tab, output]