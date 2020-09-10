import numpy as np
import matplotlib.pyplot as plt
import matplotlib


class set_style():

    def __init__(self):
        # fonts' path
        self.f = './Fonts/Montserrat-Semi<bold.ttf'

    def set_general_style_parameters(self):
        """Set some general style parameters for matplotlib
        plots so that they can be uniform throughout all the
        notebooks."""

        plt.figure(figsize = (10, 7))
        ax = fig.add_subplot(111)
        plt.text(fontproperties = fm.FontProperties(fname = self.f), fontsize = 18)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.ylabel(fontproperties = fm.FontProperties(fname = self.f), fontsize = 18, labelpad = 18)
        plt.xlabel(fontproperties = fm.FontProperties(fname = self.f), fontsize = 18, labelpad = 18)
        plt.axis(True)
        ax.set_facecolor((241/255.0, 247/255.0, 240/255.0))

        return plt, ax


























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
