import matplotlib.pyplot as plt
import numpy as np 

def plot_learning_curve(x, scores, figure_file):
    plt.plot(x, scores)
    plt.title('Running average of reward per episode')
    plt.savefig(figure_file)
