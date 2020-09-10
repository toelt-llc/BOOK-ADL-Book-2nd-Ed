import matplotlib as mpl
import matplotlib.pyplot as plt


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


























  def __init__(self, x, y, x_label, y_label, xlim, ylim, plot_type):
      self.x = x # x axis
      self.y = y # y axis
      self.x_label = x_label
      self.y_label = y_label
      self.xlim = xlim
      self.ylim = ylim
      self.plot_type = plot_type

  def create_im(self):
      """Create a simple plot"""
      f = plt.figure(figsize = (x_size, y_size))
      gs = f.add_gridspec(x_grid, y_grid)

      with sns.axes_style("whitegrid"):
          if self.plot_type == 'plot':
              plt.plot(self.x, self.y, ls = ls, color = color, linewidth = lw)
          elif self.plot_type == 'scatter':
              plt.scatter(self.x, self.y, marker = '.', c = color)
          plt.xlabel(self.x_label, fontsize = fs_lab)
          plt.ylabel(self.y_label, fontsize = fs_lab)
          plt.xlim(self.xlim[0], self.xlim[1])
          plt.ylim(self.ylim[0], self.ylim[1])
          plt.xticks(fontsize = fs_tick)
          plt.xticks(fontsize = fs_tick)

      f.tight_layout()
      return plt

  def add_x(self, plt):
      """Add f(x) = x"""
      plt.plot(self.xlim, self.xlim, ls = ls, color = color, linewidth = lw)
      return plt
