from .tilebased.heuristic import Constraint
from typing import TextIO, Dict, Callable
import re

blif_gate = re.compile(r'.gate (\w+) (.+)')
blif_pins = re.compile(r'(\w+)=(\w+)')

def chimera_map_constraint(pins, gate_data, ancilla_accum):
    #alpha order for old format
    pins_names = ["A", "B", "C", "D", "E", "F", "G", "H"]


    #first half is top row, second half is bottom
    dim = gate_data["dims"][2]
    allpos = range(1, dim*2+1)
    posmap = {x:0 for x in allpos[:dim]}
    posmap.update({y:1 for y in allpos[dim:]})

    retg = [[], []]
    for pin, pos in zip(pins_names, gate_data["vP"][0][0]):
        pin = pin[0]
        assert isinstance(pin, int), ("error parsing genlib", pin)
        retg[posmap[pos]].append(pins[pin])
        del posmap[pos]

    #leftover posmap are ancilla
    for to in posmap.values():
        retg[to] = "anc_" + str(ancilla_accum)
        ancilla_accum += 1

    return Constraint(retg), ancilla_accum

def read_blif(f, db, map_constraint):
    # type: (TextIO, Dict, Callable) -> [Constraint]
    ret = []
    ancilla_accum = 1
    for line in f:
        if not line.startswith(".gate"):
            continue
        gate_name, pins = blif_gate.match(line).groups()
        pins = {pin:var for pin, var in  blif_pins.findall(pins)}

        gate_data = db[gate_name]

        constr, ancilla_accum = map_constraint(pins, gate_data, ancilla_accum)
        ret.append(constr)

    return ret
