"""
Unit dimensioning checks
Copyright 2016, Matthew Fulghum [mfulghum]

9 June 2016 - Initial commit.
10 June 2016 - Improved speed of the dimension checks
"""

import sqlite3
import re
import numpy as np
import os

db = sqlite3.connect(os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/units.db'))
cursor = db.cursor()

unit_regex = re.compile('([a-zA-Z]+)(?:\^(-?[0-9]+)|)')

unit_query = '''
    SELECT
      TRIM(prefix) || TRIM(symbol) AS unit,
      units.power + prefixes.power AS power,
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
    GROUP BY unit
    '''

def check_dimensionality(unit_string):
    units = [{'unit':unit_power[0], 'power':(int(unit_power[1]) if unit_power[1] else 1) * (-1 if side else 1)}
             for side,units in enumerate(unit_string.strip().split('/')) for unit_power in unit_regex.findall(units)]
    if not units:
        return 0, np.zeros(7, dtype=np.int8)
    
    dimensionality = np.sum([np.array(cursor.execute(unit_query, unit).fetchone()[1:]) * unit['power'] for unit in units], axis=0).astype(np.int8)
    return dimensionality[0], dimensionality[1:]
