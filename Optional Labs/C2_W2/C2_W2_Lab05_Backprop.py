from sympy import *
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button
import ipywidgets as widgets
from lab_utils_backprop import *

plt.close("all")
plt_network(config_nw0, "./images/C2_W2_BP_network0.PNG")

w = 3
a = 2+3*w
J = a**2
print(f"a = {a}, J = {J}")

a_epsilon = a + 0.001       # a epsilon
J_epsilon = a_epsilon**2    # J_epsilon
k = (J_epsilon - J)/0.001   # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_da ~= k = {k} ")

sw,sJ,sa = symbols('w,J,a')
sJ = sa**2
sJ
sJ.subs([(sa,a)])
dJ_da = diff(sJ, sa)
dJ_da

w_epsilon = w + 0.001       # a  plus a small value, epsilon
a_epsilon = 2 + 3*w_epsilon
k = (a_epsilon - a)/0.001   # difference divided by epsilon
print(f"a = {a}, a_epsilon = {a_epsilon}, da_dw ~= k = {k} ")

sa = 2 + 3*sw
sa
da_dw = diff(sa,sw)
da_dw

dJ_dw = da_dw * dJ_da
dJ_dw

w_epsilon = w + 0.001
a_epsilon = 2 + 3*w_epsilon
J_epsilon = a_epsilon**2
k = (J_epsilon - J)/0.001   # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")

plt.close("all")
plt_network(config_nw1, "./images/C2_W2_BP_network1.PNG")

# Inputs and parameters
x = 2
w = -2
b = 8
y = 1
# calculate per step values   
c = w * x
a = c + b
d = a - y
J = d**2/2
print(f"J={J}, d={d}, a={a}, c={c}")