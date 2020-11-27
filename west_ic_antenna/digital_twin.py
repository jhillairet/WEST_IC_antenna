# -*- coding: utf-8 -*-
"""
WEST ICRH Antenna RF Model Digital Twin

@author: Julien Hillairet

This class shall be used in the frame of a jupyter notebook
in order to get the interactive functionalities

"""
import ipywidgets as widgets
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from . antenna import WestIcrhAntenna

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
        C3 = widgets.FloatSlider(value=120, min=30.0, max=120.0, step=1e-4,
                                 continuous_update=False)
        C4 = widgets.FloatSlider(value=120, min=30.0, max=120.0, step=1e-4,
                                 continuous_update=False)
        capas = widgets.HBox([widgets.VBox([widgets.Box([widgets.Label('C1: '), C1]),
                                            widgets.Box([widgets.Label('C2: '), C2])]),
                              widgets.VBox([widgets.Box([widgets.Label('C3: '), C3]),
                                            widgets.Box([widgets.Label('C4: '), C4])])
                             ])

        # excitation widgets
        side = widgets.Dropdown(options=[('Left Side only', 1), ('Right Side only', 2), ('Both Sides', 3)], value=1, description='Side:')
        phase_rel = widgets.FloatSlider(value=180, min=0, max=360, step=1,
                                        continuous_update=False)
        excitation = widgets.VBox([
            widgets.Box([widgets.Label('Power from:'), side]),
            widgets.Box([widgets.Label('Phase: '), phase_rel])
            ])

        # matching widgets
        match_freq = widgets.FloatSlider(min=40, max=65, value=55.5, step=1e-1,
                                        continuous_update=False)
        match_type = widgets.RadioButtons(options=['Left Side only', 'Right Side only', 'Both Sides Separately'])
        match_sol = widgets.RadioButtons(options=['1', '2'])
        match_button = widgets.Button(description='Match', disabled=False,
                               button_style='', tooltip='Click me', icon='check')
        out_matching = widgets.Output()

        matching = widgets.VBox([
            widgets.Box([widgets.Label('Match frequency:'), match_freq]),
            widgets.Box([widgets.Label('Solution type:'), match_sol]),
            match_type,
            match_button,
            out_matching
        ])

        # frequency widgets
        frequency_range = widgets.FloatRangeSlider(
            value=[50, 60],
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
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8,4));
        [a.set_ylim(-30, 2) for a in axes]
        [a.grid(True) for a in axes]
        axes[0].set_ylabel('$S_{ii}$')
        axes[1].set_ylabel('$S_{act,i}$')
        axes[1].set_xlabel('Frequency [MHz]')
        fig.subplots_adjust(hspace=0)

        antenna = WestIcrhAntenna()

        def init_antenna(_):
            # Create antenna object for the given frequency span and number of points
            freq = rf.Frequency(start=frequency_range.value[0],
                                stop=frequency_range.value[1],
                                npoints=frequency_npoints.value,
                                unit='MHz')
            return WestIcrhAntenna(frequency=freq)

        def plot_s(change):
            antenna = init_antenna([])
            antenna.Cs = [C1.value, C2.value, C3.value, C4.value]
            if side.value == 1:
                power = [1, 0]
            elif side.value == 2:
                power = [0, 1]
            else:
                power = [1, 1]
            phase = [0, np.deg2rad(phase_rel.value)]

            # Ignore Division per zero warning occuring in s_act
            with np.errstate(divide='ignore'):
                s_act = antenna.s_act(power, phase)
                s = antenna.circuit().s_external

            """Remove old lines from plot and plot new ones"""
            [l.remove() for l in axes[0].lines]
            [l.remove() for l in axes[1].lines]
            axes[0].plot(antenna.f_scaled, 20*np.log10(np.abs(s[:,0,0])), color='C0')
            axes[0].plot(antenna.f_scaled, 20*np.log10(np.abs(s[:,1,1])), color='C1')
            axes[1].plot(antenna.f_scaled, 20*np.log10(np.abs(s_act[:,0])), color='C0')
            axes[1].plot(antenna.f_scaled, 20*np.log10(np.abs(s_act[:,1])), color='C1')
            [a.axvline(match_freq.value, ls='--', color='k') for a in axes]
            axes[0].set_xlim(left=frequency_range.value[0], right=frequency_range.value[1])

        # match the antenna
        def match(_):
            with out_matching:
                # what happens when we press the button
                clear_output()
                Cs = [C1.value, C2.value, C3.value, C4.value]
                if match_type.value == 'Left Side only':
                    print('Searching for a match point left side')
                    Cs_sol = antenna.match_one_side(f_match=match_freq.value*1e6,
                                                    side='left', solution_number=int(match_sol.value))
                    Cs[0] = Cs_sol[0]
                    Cs[1] = Cs_sol[1]
                    C1.value = Cs[0]
                    C2.value = Cs[1]
                elif match_type.value == 'Right Side only':
                    print('Searching for a match point right side')
                    Cs_sol = antenna.match_one_side(f_match=match_freq.value*1e6, side='right', solution_number=int(match_sol.value))
                    Cs[2] = Cs_sol[2]
                    Cs[3] = Cs_sol[3]
                    C3.value = Cs[2]
                    C4.value = Cs[3]
                elif match_type.value == 'Both Sides Separately':
                    print('Searching for a match point both sides separately')
                    Cs_sol = antenna.match_both_sides_separately(f_match=match_freq.value*1e6, solution_number=int(match_sol.value))
                    Cs = Cs_sol
                    C1.value = Cs[0]
                    C2.value = Cs[1]
                    C3.value = Cs[2]
                    C4.value = Cs[3]

                print(Cs)
                plot_s([])

        # Define callbacks
        match_button.on_click(match)
        C1.observe(plot_s)
        C2.observe(plot_s)
        C3.observe(plot_s)
        C4.observe(plot_s)
        side.observe(plot_s)
        phase_rel.observe(plot_s)
        frequency_range.observe(plot_s)
        frequency_npoints.observe(plot_s)

        # setting the tab windows
        tab = widgets.Tab()
        tab.children = [capas, excitation, matching, frequency]
        tab.set_title(0, 'Capacitors')
        tab.set_title(1, 'Excitation')
        tab.set_title(2, 'Matching')
        tab.set_title(3, 'Frequency Range')

        plot_s([])  # show something at the startup
        # tab  # must be last
        self.children = [tab, output]