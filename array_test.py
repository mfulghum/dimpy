# -*- coding: utf-8 -*-
"""
Created on Sun Jun 05 17:56:27 2016

@author: Matthew Fulghum
"""

import numpy as np
import dimpy

vel = dimpy.data([1.0, 2.0, 3.0], units='mm/s')
area = dimpy.data([20.0], units='m^2')

mass = dimpy.data([1.0], units='kg')
acceleration = dimpy.data([1.0], units='m/s^2')
force = dimpy.data([1.0], units='N')

length = dimpy.data([1.0], units='cm')
volume = dimpy.data([1.0], units='mL')

# Pendulum model
T = dimpy.unit('s')
M = dimpy.unit('kg')
L = dimpy.unit('m')
g = dimpy.unit('m/s^2')

variables = [T, M, L, g]
print(dimpy.analyze(variables))

# Black body model
u = dimpy.unit('kg/(m^2-s^2)')
wavelength = dimpy.unit('m')
h = dimpy.unit('kg-m^2/s')
c = dimpy.unit('m/s')
kB = dimpy.unit('kg-m^2/(s^2-K)')
T = dimpy.unit('K')

variables = [u, wavelength, h, c, kB, T]
print(dimpy.analyze(variables))

