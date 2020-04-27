import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

widgets.IntSlider(
    min=0,
    max=10,
    step=1,
    description='Slider:',
    value=3
)
slider = widgets.FloatLogSlider(min=-10,max=3)
display(slider)
btn = widgets.Button(description='Medium')
display(btn)
output = widgets.Output()
plot_output = widgets.Output()
display(output)
display(plot_output)
xpos = np.arange(0,100)/100.0
def btn_eventhandler(obj):
    output.clear_output()
    plot_output.clear_output()
    with output:
        print('Slider is '+str(slider.value))
    with plot_output:
        plt.figure('Hi there')
        plt.plot(np.exp(xpos * slider.value))
        plt.show()
btn.on_click(btn_eventhandler)
