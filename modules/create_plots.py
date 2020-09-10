import seaborn as sns
import matplotlib.pyplot as plt

# style settings
x_size = 10
y_size = 7
x_grid = 5
y_grid = 2
fs_lab = 16
ls = 'solid'
color = 'black'
lw = 2
fs_tick = 12

class set_style():

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
