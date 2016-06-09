# -*- coding: utf-8 -*-
"""
Created on Sun Jun 05 17:56:27 2016

@author: Matthew Fulghum
"""

import dimpy
import re

vel = dimpy.array([1.0, 2.0, 3.0], units='m/s')

unit_string = 'm^2/s'

unit_regex = re.compile('([a-zA-Z]+)(?:\^(-?[0-9]+)|)')
units = [{'unit':unit_power[0], 'power':(int(unit_power[1]) if unit_power[1] else 1) * (-1 if side else 1)} for side,units in enumerate(unit_string.split('/')) for unit_power in unit_regex.findall(units)]

import sqlite3

db = sqlite3.connect('dimpy/units.db')
cursor = db.cursor()

unit_regex = re.compile('([a-zA-Z]+)(?:\^(-?[0-9]+)|)')

def check_dimensionality(unit_string):
    units = [{'unit':unit_power[0], 'power':(int(unit_power[1]) if unit_power[1] else 1) * (-1 if side else 1)}
             for side,units in enumerate(unit_string.split('/')) for unit_power in unit_regex.findall(units)]

    query = '''
        SELECT
          CASE WHEN prefix=' ' THEN '' ELSE prefix END || symbol AS unit,
          units.power + prefixes.power,
          units.length,
          units.mass,
          units.time,
          units.current,
          units.temperature,
          units.amount_of_substance,
          units.luminous_intensity
        FROM units
        JOIN prefixes
        WHERE unit=:unit
        '''
    dimensionality = np.sum([np.array(cursor.execute(query, unit).fetchone()[1:]) * unit['power'] for unit in units], axis=0)
    return dimensionality[0], dimensionality[1:]

