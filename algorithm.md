
# Placeandroute: Brief explanation

## Introduction:

This document describes how to get from a list of Constraints in QUBO form
to a DW-ready problem. This is done by placing the constraints somewhere on the
graph and connecting the equivalent nodes with chains (_place&route_).

The code uses an heuristic approach: we try a particular placement/routing, we
evaluate it and we try to improve it. We temporarily allow but penalize
over-allocation of qubits. We then try to reduce the overall over-allocation to
0.

I start with an explanation of the SoA of the routing problem, called Bonn
routing. This is because my solution to the placement problem relies heavily on
it.

I then describe how i generate good placements and improve them. While there are
various complex placement algorithms that do not rely on routing, I created one
that relies heavily on Bonn routing. I describe also the criteria to modify the 
placement of a single constraint (_rip&reroute_).

In part 3 I describe how I tried to keep the code flexible, by allowing multiple
approaches, and how I handle work parallelization. 

## Part 1: Bonn routing via Max/Min fair resource sharing
**Reference:** _BonnRoute: Algorithms and data structures for fast and good VLSI routing_

#### Problem
##### Given:
- A CSP problem in factor graph form $P(C \cup V, E_P)$ where $C$ is the set of
constraints ad $V$ is the set of vars

- An hardware graph, $G(Q,E_G)$ where $Q$ are qubits and $E_G$ are qubit 
couplings

- A map $placement : C \rightarrow \{Q\}$ that describes the current placement/ 
qubits occupied by constraints

- A map $terminals : V \rightarrow \{Q\}$ that maps variables to hardware qubits


##### Find:
- A map $chain : V \rightarrow \{Q\}$ that maps each CSP variable to a tree
connecting all its terminal

##### Constrained by:
- No overlapping between chains and between constraints

$$\forall c_1,c_2 \in C,\quad placement(c_1) \cap  placement(c_2) = \emptyset$$
$$\forall v_1,v_2 \in V,\quad chain(v_1) \cap  chain(v_2) = \emptyset$$
$$\forall v_1 \in V,c_2 \in C\quad chain(v_1) \cap  placement(c_2) = \emptyset$$

#### Algorithm

##### Graph representation
- We work on a tile graph $G'$ where each vertex represents a group
of identical qbits, (i.e a row of a chimera tile). In this case we define also
two functions $capacity: Q' \rightarrow \mathbb{N}$ and 
$usage: Q' \rightarrow \mathbb{N}$ to define how many qubits have been used and
require $usage(q) \le capacity(q)$. Converting the $chains$ from $Q'$ to $Q$ is 
not trivial but annealing seems to work well. 

- In order to reduce complexty  each tile is represented by two vertices of capacity 4.
Possible placement choices are stored in a list of (q_1, q_2) tuples (to avoid
placing a constraint inbetween two tiles). A tuple (q_2, q_1) in the choice
list represents the same tile in the opposite orientation.


##### Bonn routing
- From the paper, we can use an algorithm that approximates the minimum Steiner
tree to allocate fairly qubits between all variables $V$.

- The first step produces a convex sum (~ "weighted average") between candidate
  trees.

- In the second step we use randomization/ annealing to choose the best tree.

- Important later: Bonnroute can be used incrementally. If we have a partial map
$chains' : V \rightarrow \{Q'\}$ that connect some but not all terminals, we can
add these chains to the terminals, using 
$terminals'(v) = terminals(v) \cup chains'(v)$ as terminal.

##### On Steiner tree approximation

- There are good Steiner tree algorithm, turns out the SoA is not necessary and 
a fast algorithm is better

- The first algorithm connects through the shortest path the terminal in 
succession

- The second algorithm chooses a terminal at random, connects with the closest 
other terminal until all terminal are connected

- The second algorithm is slower, converges faster on C16 but not always

- Hypothesis: it produces shorter trees, but with higher variance, thus requires
 more calls

## Part 2: Heuristic placement
#### Goal:
- A good map $placement : C \rightarrow \{Q\}$ 

#### Main flow:
- Initialize placement randomly/greedily
- Choose badly connected constraints, rip and reroute them
- Often, recalculate all chains
- Sometimes, restart the search

#### On greedy initialization
- Traverse the CSP graph, try to place and route visited nodes
- Start from "central" constraints/qubits (betweenness seems to work)
- BFS traversal (hopefully a well placed subgraph is not rerouted anymore)

#### On rip&reroute
- **How to choose ripped constraints:** sum of chain costs, use tabu list to avoid
ripping the same constraint over and over
- **How to choose new placement:** available space, approx cost of new chains
- Fast, sometimes worsen score

#### Rerouting everything
- Slow, seems beneficial
- Rerouting effort (i.e. # of candidate trees, annealing rounds) can be slowly 
increased


## Part 3: Tactics and Parallelism

#### Rationale:

- If the best approach to take is not clear, take all possible approach
- Flexibility: we can choose different parameters/schedules for tactics
- General framework: Initialize the placement, then improve it
- Initialize starts from an empty placement, improve from a full placement

#### Combinators:
- A tactic that combines other tactics
- Example: RepeatTactic, repeats a tactic until there is no improvement


#### Naive parallelism:

- simple parallelism: Initialize and improve in parallel, pick best final result
- Easy to implement on top of current code
- uses Python's `multiprocessing` module

## Appendix: notes


- Plugging different CSP libraries is easy, as Constraints objects contain only a
simple description of the variables to be connected.

- What I called annealing in the code is various forms of stochastic search
that I wrote ad-hoc and that can probably be improved by testing, sorry for the
possible confusion

- I moved all tactics used in the heuristic placement class in a separate
module, hopefully making them easier to compose.

- Using incremental Bonnroute while rerouting everything every few turns was an
afterthought, as I used to reroute everything every time. Seems that trying to
reroute everithing after changing slightly the placement is an effective move.