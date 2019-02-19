from dwave_networkx import chimera_graph
from placeandroute.tilebased.chimera_tiles import chimera_tiles

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
    allpos = list(range(1, dim*2+1))
    posmap = {x:0 for x in allpos[:dim]}
    posmap.update({y:1 for y in allpos[dim:]})
    ordered_wires = []
    ow = {}

    retg = [[], []]
    for pin, pos in zip(pins_names, gate_data["vP"][0]):
        pin = pin[0]
        pos = pos[0]
        assert isinstance(pos, int), ("error parsing genlib", pin)
        retg[posmap[pos]].append(pins[pin])
        ow[pos] = pins[pin]
        del posmap[pos]

    #leftover posmap are ancilla
    for apos,to in posmap.items():
        ancilla_name = "anc_" + str(ancilla_accum)
        retg[to].append(ancilla_name)
        ow[apos] = ancilla_name
        ancilla_accum += 1

    for pos in range(1, dim*2+1):
        ordered_wires.append(ow[pos])


    return Constraint(retg), ordered_wires, ancilla_accum

def chimera_map_constraint2(pins, gate_data, ancilla_accum):
    pin_labels = ["O", "I0", "I1", "I2", "I3", "I4", "I5", "I6"]
    reference_graph = chimera_tiles(1)
    posmap = {node: i for i, nodeset in enumerate(reference_graph.nodes())
                                for node in nodeset}
    retg = [[], []]

    for pin_name, wire_name in pins:
        pos_name = "pos"

    ordered_wires = [pins[pin_name] for pin_name in pin_labels]
    return Constraint(), ordered_wires, ancilla_accum

def read_blif(f):
    # type: (TextIO) -> List[Tuple[str, Dict]]
    ret = []
    for line in f:
        if not line.startswith(".gate"):
            continue
        gate_name, pins = blif_gate.match(line).groups()
        pins = {pin:var for pin, var in  blif_pins.findall(pins)}
        ret.append((gate_name, pins))
    return ret

def blif_to_constraints(blif, db, map_constraint):
    # type: (List[Tuple[str, Dict]], Dict, Callable) -> List[Constraint]
    ret = [] #type: List[Constraint]
    ancilla_accum = 1
    for gate_name, pins in blif:

        gate_data = db[gate_name]

        constr, wires, ancilla_accum = map_constraint(pins, gate_data, ancilla_accum)
        constr.wires = wires
        ret.append(constr)


    return ret
