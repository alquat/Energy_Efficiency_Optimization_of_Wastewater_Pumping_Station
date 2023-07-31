import numpy as np
import matplotlib.pyplot as plt

class PlotPumpData:
    def __init__(self, x, y, num_bins, r_squared, mse, predictions, lower_bound, upper_bound, title, x_label_plt1='', y_label_plt1='',x_label_plt2='', y_label_plt2='' ):
        self.x = x
        self.y = y
        self.num_bins = num_bins
        self.r_squared = r_squared
        self.mse = mse
        self.pred = predictions
        self.title = title
        self.x_label_plt1 = x_label_plt1
        self.y_label_plt1 = y_label_plt1
        self.x_label_plt2 = x_label_plt2
        self.y_label_plt2 = y_label_plt2
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def plot_data(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].scatter(self.x, self.y, alpha=0.4, color='blue')
        axes[0].scatter(self.x, self.pred, color='purple')
        axes[0].set_xlabel(self.x_label_plt1)
        axes[0].set_ylabel(self.y_label_plt1)
        axes[0].set_title(f'R^2: {self.r_squared:.4f} MSE: {self.mse:.4f}')
        axes[0].set_xlim(self.lower_bound, self.upper_bound)
        axes[0].grid(alpha=0.5)
        
        counts, bins, _ = axes[1].hist(self.y, self.num_bins, density=True, alpha=0.5)
        axes[1].set_xlabel(self.x_label_plt2)
        axes[1].set_ylabel(self.y_label_plt2)
        axes[1].set_title(f'{np.mean(self.pred):.4f} ± {np.std(self.pred):.4f}')
        axes[1].grid(axis='y', alpha=0.5)
        
        fig.suptitle(self.title, fontsize=16, fontweight='bold')
        fig.subplots_adjust(top=0.75)
        fig.tight_layout()
        
        plt.show()
