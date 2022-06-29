import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from matplotlib import cycler

plt.ion()

def scores_chart(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.xlabel('Games')
    plt.ylabel('Score: ')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
    plt.grid(color='w', linestyle='solid')
    colors = cycler('color',
                    ['#EE0000', '#000000', '#00EE00',
                     '#EECC55', '#88BB44', '#FFBBBB'])
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
           axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('patch', edgecolor='#E6E6E6')
    plt.rc('lines', linewidth=2, linestyle=':')


def comparison_chart(trained_model_average_score, new_model_average_score):
    models = ('Trained Model', 'New model')
    y_pos = np.arange(len(models))
    performance = [trained_model_average_score, new_model_average_score]

    display.display(plt.gcf())
    display.clear_output()

    plt.figure()
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, models)
    plt.ylabel('Average score')
    plt.title('Average score by model')
    plt.show(block=True)
