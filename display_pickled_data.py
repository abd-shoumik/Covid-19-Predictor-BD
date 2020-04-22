#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:32:34 2020

@author: shoumik
"""



import numpy as np
import matplotlib.pyplot as plt
import pickle 

#loading the dumped files
loaded_cases=pickle.load(open('model_cases.pkl','rb'))
loaded_deaths=pickle.load(open('model_deaths.pkl','rb'))
cases_pred_vsl=pickle.load(open('cases_figure.pkl','rb'))
deaths_pred_vsl=pickle.load(open('deaths_figure.pkl','rb'))

#function for displaying figures
def show_figure(fig):

    # create a dummy figure and use its
    # manager to display "fig"  
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

    show_figure(cases_pred_vsl)
    show_figure(deaths_pred_vsl)

#Visualize your pickled figure
deaths_pred_vsl.show()
cases_pred_vsl.show()
