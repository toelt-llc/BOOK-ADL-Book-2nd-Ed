import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class set_style():

    def __init__(self):
        # fonts' path
        self.f = './Fonts/Montserrat-SemiBold.ttf'

    def set_general_style_parameters(self):
        """Set some general style parameters for matplotlib
        plots so that they can be uniform throughout all the
        notebooks."""

        mpl.rcParams['figure.figsize'] = [10, 7]
        mpl.rcParams['figure.dpi'] = 300

        mpl.rcParams['font.size'] = 18
        mpl.rcParams['legend.fontsize'] = 'large'
        mpl.rcParams['figure.titlesize'] = 'medium'

        mpl.rcParam['lines.markersize'] = np.sqrt(20)

        mpl.rcParams['lines.linewidth'] = 2
        mpl.rcParams['lines.dashed_pattern'] = [6, 6]
        mpl.rcParams['lines.dashdot_pattern'] = [3, 5, 1, 5]
        mpl.rcParams['lines.dotted_pattern'] = [1, 3]
        mpl.rcParams['lines.scale_dashes'] = False

        mpl.rcParams['legend.fancybox'] = False
        mpl.rcParams['legend.loc'] = 'best'
        mpl.rcParams['legend.numpoints'] = 2
        mpl.rcParams['legend.fontsize'] = 'large'
        mpl.rcParams['legend.framealpha'] = None
        mpl.rcParams['legend.scatterpoints'] = 3
        mpl.rcParams['legend.edgecolor'] = 'inherit'

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor((241/255.0, 247/255.0, 240/255.0))

        return fig, ax
