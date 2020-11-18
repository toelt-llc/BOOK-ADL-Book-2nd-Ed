import matplotlib as mpl
import numpy as np


class set_style():

    def __init__(self):
        # fonts' path
        self.f = 'ADL-Book-2nd-Ed/modules/Fonts/Montserrat-Medium.ttf'

    def set_general_style_parameters(self):
        """Set some general style parameters for matplotlib
        plots so that they can be uniform throughout all the
        notebooks."""

        mpl.rcParams['figure.figsize'] = [8, 5] # dimensions
        mpl.rcParams['figure.dpi'] = 80 # resolution
        mpl.rcParams['figure.titlesize'] = 'medium' # title dimension

        mpl.rcParams['font.size'] = 18 # font

        mpl.rcParams['lines.linewidth'] = 2 # line width

        mpl.rcParams['legend.loc'] = 'best' # legend position inside plot
        mpl.rcParams['legend.fontsize'] = 'medium' # legend font

        mpl.rcParams['axes.facecolor'] = (241 / 255.0, 247 / 255.0, 240 / 255.0)  # background color

        return self.f