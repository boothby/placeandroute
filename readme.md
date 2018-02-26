
## Brief explanation of the algorithm

### Part 1: Bonn routing via Max/Min fair resource sharing
#### Reference
BonnRoute: Algorithms and data structures for fast and good VLSI routing

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

##### Notes:
- It is easier to work on a tile graph $G'$ where each vertex represents a group
of identical qbits, (i.e a row of a chimera tile). In this case we define also
two functions $capacity: Q' \rightarrow \mathbb{N}$ and 
$usage: Q' \rightarrow \mathbb{N}$ to define how many qubits have been used and
require $usage(q) \le capacity(q)$. Converting the $chains$ from $Q'$ to $Q$ is 
not trivial but annealing seems to work well. 

- From the paper, we can use an algorithm that approximates the minimum Steiner
tree to allocate fairly qubits between all variables $V$.

- The first step produces a convex sum (~ "weighted average") between candidate
trees, we use randomization/ annealing to choose the best tree.

- Important later: Bonnroute can be used incrementally. If we have a partial map
$chains' : V \rightarrow \{Q'\}$ that connect some but not all terminals, we can
add these chains to the terminals, using 
$terminals'(v) = terminals(v) \cup chains'(v)$ as terminal.


### Part 2: Heuristic placement
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
- How to choose ripped constraint: sum of chain costs, use tabu list to avoid
ripping the same constraint over and over
- How to choose new placement: available space, approx cost of new chains
- Fast, sometimes worsen score

#### Rerouting everything
- Slow, seems beneficial
- Rerouting effort (i.e. # of candidate trees, annealing rounds) can be slowly 
increased


### Notes

- For Chimera graphs, each tile is represented by two vertices of capacity 4.
Possible placement choices are stored in a list of (q_1, q_2) tuples (to avoid
placing a constraint inbetween two tiles). A tuple (q_2, q_1) in the choice
list represents the same tile in the opposite orientation.

- Plugging different CSP libraries is easy, as Constraints objects contain only a
simple description of the variables to be connected.

- What I called annealing in the code is various forms of stochastic search
that I wrote ad-hoc and that can probably be improved by testing, sorry for the
possible confusion

- I moved all tactics used in the heuristic placement class in a separate
module, hopefully making them easier to compose.