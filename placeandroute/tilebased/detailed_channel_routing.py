"""
Build Chimera routes from tile routes.
Author: Aidan Roy
"""
import networkx as nx
import random
import itertools
import math
import copy

def linear_index_to_chimera(linear_indices, m, n=None, t=None):
    """
    [[i0, j0, u0, k0], [i1, j1, u1, k1], ...] = linear_index_to_chimera(linear_indices, m, n, t)

    Convert linear indices to indices (i, j, u, k) into the Chimera adjacency
    matrix. m, n, t give the size of the Chimera graph. The lattice is m x n
    and each bipartite component is K_{t, t}. i, j, u, k may be column
    vectors of a common length and we return the linear index for each
    element. i, j label the lattice location with 0 <= i <= m - 1 and 0 <= j <= n - 1,
    0 <= u <= 1 labels the partition, and 0 <= k <= t - 1 the vertex within a
    partition. See chimera_to_linear_index for the inverse function.

    Args:
        linear_indices: a list or tuple of linear indices
        m: m for Chimera graph
        n: n for Chimera graph
        t: t for Chimera graph

    Returns:
        A list of list, each list contained has four elements representing
        i, j, u, k

    Raises:
        ValueError: bad parameter value
    """
    if n is None:
        n = m
    if t is None:
        t = 4

    hd = 2 * t
    vd = n * hd
    chimera_indices = []
    for ti in linear_indices:
        r, ti = divmod(ti, vd)
        c, ti = divmod(ti, hd)
        u, k = divmod(ti, t)
        chimera_indices.append([r, c, u, k])
    return chimera_indices


def greedy_list_coloring(adjacency, valid_colors):
    # list-colouring problem: assign a valid color to each
    # vertex so that neighbours in the graph get different colors
    #
    # greedy algorithm:
    # pick a variable with a minimal number of valid choices, assign it.
    #
    # adjacency: dictionary of neighbors
    # valid_colors: dictionary of the set of valid colors for each node.
    #   valid_colors is destroyed by this function.

    colors = {v: None for v in adjacency}
    while valid_colors:

        # pick vertex v with a minimal number of valid colors
        # TODO: and a maximal number of neighbours
        (v, v_colors) = min(valid_colors.items(), key=lambda x: len(x[1]))
        if len(v_colors) == 0:
            # coloring failed.
            return None

        # pick a random valid color for v
        c = random.choice(tuple(v_colors))
        colors[v] = c
        del valid_colors[v]

        # disallow that color for uncolored neighbours
        # "discard" is like "remove" but does not raise an error if the element
        # does not exist
        for n in adjacency[v]:
            if colors[n] is None:
                valid_colors[n].discard(c)

    return colors


def solve_channel(channel_graph, quotient_embedding, num_tracks, working_tracks,
                  missing_edges=[],
                  bad_track_assignments=[]):
    """
    Given an embedding in a channel (list of quotient qubits for each
    variable), assign variables to tracks in such a way that variables on
    on the same quotient qubit get different tracks.

    In the context of a Chimera graph: we have indices (i,j,k,l) = (row,col,
    orientation,qubit). A channel is a row of horizontal qubits (i,*,1,*)
    or a column of vertical qubits (*,j,0,*). A quotient qubit has the form
    (i,j,k) and a track is a qubit index l.

    Args:
        quotient_embedding: dict()
        quotient_embedding[v] is the list of quotient vertices in the
        channel on which variable v is embedded

        num_tracks: int
        num_tracks is the total number of available tracks. Tracks are
        assumed to be labelled 0,1,...,num_tracks-1.

        working_tracks: dict()
        working_tracks[q] is list of valid tracks (i.e.
        working qubits) for quotient vertex q

        missing_edges: list of 3-tuples, default = []
        rows have the form (i,j,k), where (i,j) is an edge in the
        quotient graph and k is the missing track. If x is assigned to both i
        and j, don't use track k.

        bad_track_assignments: list of 2-tuples, default = []
        has the form (x,k), where x is a variable and k
        is a track. x should not be assigned track k. These restrictions come
        from missing couplers between channels.

    Returns:
        track_assignments: dict(), or None
        track_assignments[v] is the track for variable v.
        if no assignment is found, returns None.

    This is a graph coloring problem for the variable conflict graph
    (the graph of which variables cannot be assigned to the same track).
    There are many potential ways to solve it. However, most track algorithms
    don't deal with missing tracks (qubits), so we need our own method.

    """

    # number of tracks
    all_tracks = set(range(num_tracks))

    def intersection(a,b):
        # not sure if this is the best way to do this
        return set(a).intersection(set(b))


    # generate a list of track assignments we care about
    # if a quotient chain leaves the channel and then returns, the
    # separate components can be assigned different tracks.
    component_embedding = dict()
    component_var = dict()
    count = 0
    for u, quotient_chain in quotient_embedding.items():
        quotient_channel_graph = nx.subgraph(channel_graph, quotient_chain)
        components = nx.connected_components(quotient_channel_graph)
        for c in components:
            component_embedding[count] = quotient_channel_graph.subgraph(c)
            component_var[count] = u
            count += 1


    # generate a conflict graph
    # (graph of variables with conflicting embeddings)
    conflicts = []
    for u, v in itertools.combinations(component_embedding.keys(), 2):
        if intersection(component_embedding[u], component_embedding[v]):
            conflicts.append((u,v))
    conflict_graph = nx.Graph(conflicts)
    # make sure all variables are included, even if no conflicts:
    for v in component_embedding:
        conflict_graph.add_node(v)

    # generate a list of variables used by each quotient vertex
    quotient_variables = {b: [] for b in working_tracks}
    for v, quotient_chain in component_embedding.items():
        for b in quotient_chain:
            quotient_variables[b].append(v)

    # sanity check channel capacity for each quotient vertex
    # (check there are enough working tracks for the variables)
    for b in working_tracks:
        if len(quotient_variables[b]) > len(working_tracks[b]):
            raise ValueError('Channel capacity not large enough.\n')

    # write down the missing tracks for each quotient vertex
    missing_tracks = {b: all_tracks.difference(set(qubits)) for b,
        qubits in working_tracks.items()}

    # compute valid track assignments for each variable
    valid_tracks = {v: copy.deepcopy(all_tracks) for v in component_embedding}
    for v, quotient_chain in component_embedding.items():
        for b in quotient_chain:
            for t in missing_tracks[b]:
                valid_tracks[v].discard(t)

    # remove variable assignments based on missing edges
    # missing_edges has the form (i,j,k), where (i,j) is an edge in the
    # channel and k is the missing track. If x is assigned to both i
    # and j, don't use track k.
    for (b_i, b_j, track) in missing_edges:
        bad_variables = intersection(quotient_variables[b_i], quotient_variables[b_j])
        for v in bad_variables:
            valid_tracks[v].discard(track)

    # remove bad track assignments
    # has the form (x,k), where x is a variable and k
    # is track. x should not be assigned track k.
    for (v, track) in bad_track_assignments:
        valid_tracks[v].discard(track)

    # now we have a list-colouring problem: assign a valid track to each
    # variable so that neighbours in the conflict graph get a different
    # track.
    component_assignments = greedy_list_coloring(conflict_graph, valid_tracks)

    track_assignments = {v: dict() for v in quotient_embedding.keys()}
    for c, component_chain in component_embedding.items():
        v = component_var[c]
        t = component_assignments[c]
        track_assignments[v].update({b: t for b in component_chain})

    return track_assignments


def get_chimera_dims(A):
    return int(math.sqrt(len(A)/8))


def get_chimera_quotient_graph(chimera_graph):
    # reduced chimera graph consists of identified L-sets of qubits
    N = get_chimera_dims(chimera_graph)
    L = 4

    def node_data(S):
        # node data for the set S of chimera vertices:
        # add the first three chimera indices of S.
        u = list(S)[0]
        [[i, j, k, _]] = linear_index_to_chimera([u], N, N, L)
        return {'chimera_index': (i,j,k)}

    def node_relation(u,v):
        # assumes a dnx chimera graph
        index_u = chimera_graph.nodes[u]['chimera_index'][0:3]
        index_v = chimera_graph.nodes[v]['chimera_index'][0:3]
        return index_u == index_v

    # nx.quotient_graph nodes are frozensets
    quotient_graph = nx.quotient_graph(chimera_graph, node_relation,
                                  node_data=node_data)

    # quotient_mapping is the map of vertices of chimera to (quotient
    # vertex,track)
    quotient_mapping = {}

    # qubit_mapping is the map from (quotient_vertex,track) to chimera
    qubit_mapping = {}
    for q in quotient_graph:
        for v in q:
            [[_, _, _, t]] = linear_index_to_chimera([v], N, N, L)
            quotient_mapping[v] = (q,t)
            qubit_mapping[(q, t)] = v

    return quotient_graph, quotient_mapping, qubit_mapping


def get_chimera_quotient_channels(quotient_graph, L=1):
    # get channels in a shifted Chimera graph
    n = len(quotient_graph)
    N = int(math.sqrt(n/(2*L)))

    # tracks are rows and columns
    channels = {(k,i): [] for i in range(N) for k in range(2)}
    for q in quotient_graph:
        (i,j,k) = quotient_graph.nodes[q]['chimera_index']
        if k==0:
            # vertical track, use column j
            channels[(k,j)].append(q)
        else:
            # horizontal track, use row i
            channels[(k,i)].append(q)

    return channels


def detailed_channel_routing(quotient_emb, graph,
                             **options):
    """
    Converts an embedding in quotient graph into a an embedding in full
    hardware graph.
    This is a detailed routing algorithm for Chimera (or other architectures).

    We identify "channels" in the quotient graph: quotient qubits for
    which if a variable is assigned to multiple quotient qubits in the same
    track set, it must be assigned the same track (index) to all of those
    quotient qubits.
    each channel is a series of quotient qubits and corresponds to a series of
    tracks in the original hardware graph (where a track is a series of
    connected qubits).

    For chimera, each row and column is a channel and there are 4
    tracks per channel (the l-index in Chimera notation).


    Args:
        quotient_emb: dict
            quotient_emb[x] is the chain for variable x in the quotient graph

        graph: Networkx graph
            hardware graph

        options:
            'architecture': must be 'chimera' for now.

            'constraint_quotient_emb': dict, default = None

                constraint_quotient_emb[c] is the list of vertices in the quotient
                graph that encode constraint c. If it not empty,
                quotient_terminal_emb should also be specified.

            'quotient_terminal_emb': dict, default = None
                quotient_terminal_emb[x] is the list of vertices in the chain
                quotient_emb[x] which are terminals (part of a constraint).
                (Note that a vertex q in quotient_emb[x] might intersect
                constraint_quotient_emb{c}, where x is in c,
                even when x not a terminal.)

            'anneal_offset_ranges': list of lists, default = None
                available offset ranges on each qubit, from
                solver.properties.anneal_offset_ranges. offset_ranges[i] has the
                form [min_offset, max_offset] where min_offset <= 0, max_offset
                >= 0. If specified, 'ideal_quotient_offsets' should also be
                specified.

            'ideal_quotient_offsets': dict
                a dictionary mapping quoteint vertices to desired anneal
                offsets (for example from chain_offsets_nonlinear).
                If 'ideal_quotient_offsets' and 'anneal_offset_ranges' are
                specified, the algorithm will attempt to maximize the scale
                of the working anneal offset strategy by solving each
                channel several times and choosing the best answer.

    Returns:
        vertex_embedding: dict{}
            vertex_embedding[x] is the chain for variable x in
            the hardware graph "graph"

        constraint_emb: dict{}
            If constriant_quotient_emb is specified, then constraint_emb[c]
            is the list of vertices in the hardware graph that encode
            constraint c.
    """

    # We will use the following variable conventions:
    # c = channel or constraint
    # q = quotient vertex
    # v = hardware vertex
    # x = variable
    # t = track

    # Default options:
    architecture = options['architecture'] if 'architecture' in options else\
        'chimera'

    constraint_quotient_emb = options['constraint_quotient_emb'] if \
        'constraint_quotient_emb' in options else None

    quotient_terminal_emb = options['quotient_terminal_emb'] if \
        'quotient_terminal_emb' in options else None

    anneal_offset_ranges = options['anneal_offset_ranges'] if \
        'anneal_offset_ranges' in options else None

    ideal_quotient_offsets = options['ideal_quotient_offsets'] if \
        'ideal_quotient_offsets' in options else None



    # get quotient graph and channels
    # by default, architecture is 'chimera'.
    # Other architectures may be added later
    if architecture=='chimera':
        [quotient_graph, quotient_mapping, qubit_mapping] = \
            get_chimera_quotient_graph(graph)
        channels = get_chimera_quotient_channels(quotient_graph)
        num_tracks = 4
    else:
        raise ValueError('Unrecognized architecture')

    # for each quotient graph vertex, write down the variables on it
    quotient_variables = {q: [] for q in quotient_graph}
    for v, chain in quotient_emb.items():
        for q in chain:
            quotient_variables[q].append(v)

    # for each quotient graph vertex, write down the constraints associated with it
    if constraint_quotient_emb:
        quotient_constraints = {q: [] for q in quotient_graph}
        for constraint, chain in constraint_quotient_emb.items():
            for v in chain:
                quotient_constraints[v].append(constraint)

    # for quotient vertex, write down the channel associated with it:
    quotient_channel = {q: None for q in quotient_graph}
    for index, channel in channels.items():
        for v in channel:
            quotient_channel[v] = index

    # identify missing cross-channel edges
    missing_cross_edges = dict()
    for (q0, q1) in quotient_graph.edges():
        c0 = quotient_channel[q0]
        c1 = quotient_channel[q1]
        if c0 != c1:
            # for two different track sets, all connections should be available
            for x, v1 in itertools.product(q0,q1):
                if (x, v1) not in graph.edges():
                    # missing edge
                    if (c0, c1) not in missing_cross_edges:
                        missing_cross_edges[(c0, c1)] = [(x, v1)]
                    else:
                        missing_cross_edges[(c0, c1)].append((x, v1))

    # initialize the final embedding
    which_vars = {}
    vertex_embedding = {x: [] for x in quotient_emb.keys()}
    if constraint_quotient_emb:
        constraint_emb = {c: {} for c in constraint_quotient_emb}
    else:
        constraint_emb = None

    # Solve the track assignment problem on each channel.
    #
    # Note that each channel may be solved almost independently, because
    # when switching from one channel to another, couplers are available
    # between all track pairs.
    # However, missing couplers make this not true. To partially mitigate
    # this, we solve the channels in a random order.

    # this replaces for index, channel in channels.items():
    channel_indices = list(channels.keys())
    random.shuffle(channel_indices)
    for index in channel_indices:
        channel = channels[index]

        # write down the variables in the channel:
        vars = set()
        for v in channel:
            for q in quotient_variables[v]:
                vars.add(q)

        if vars:
            # find the embedding of the variables within the channel:
            channel_emb = {x: [q for q in quotient_emb[x] if q in channel]
                           for x in vars}

            # compute which channel qubits are working
            working_channel_tracks = {}
            for q in channel:
                working_channel_tracks[q] = [quotient_mapping[v][1] for v in q
                                             if len(graph[v]) > 0]

            # compute the channel edges that are missing
            channel_graph = quotient_graph.subgraph(channel) # induced subgraph
            missing_channel_edges = []
            for (q0,q1) in channel_graph.edges:
                for t in range(num_tracks):
                    v0 = qubit_mapping[(q0,t)]
                    v1 = qubit_mapping[(q1,t)]
                    if (v0, v1) not in graph.edges():
                        # missing edge
                        missing_channel_edges.append((q0, q1, t))
                #for v0, v1 in itertools.product(q0, q1):
                #    if (v0, v1) not in graph.edges():
                        # missing edge
                #        t = quotient_mapping[v0][1]
                #        missing_channel_edges.append((q0, q1, t))

            # compute the bad track assignments as a result of missing
            # couplers to existing track assignments
            bad_track_assignments = []
            for index2, channel2 in channels.items():
                if (index2, index) in missing_cross_edges:
                    for v2, v in missing_cross_edges[(index2, index)]:
                        (q2, t2) = quotient_mapping[v2]
                        (q, t) = quotient_mapping[v]

                        # chain interaction to respect:
                        # if x is embedded at v2 and there is an edge
                        # missing from v2 to v, don't embed x at v.
                        if v2 in which_vars:
                            q = which_vars[v2]
                            if q in vars and q in quotient_emb[q]:
                                bad_track_assignments.append((q, t))

                        if constraint_quotient_emb:
                            # constraint interactions to respect:
                            # if a constraint is embedded at both v and v2,
                            # then it will require edges between v and v2. So
                            # don't embed constraint where edges are missing.
                            c = quotient_constraints[q]
                            c2 = quotient_constraints[q2]
                            ci = set(c).intersection(set(c2)) # common constraints
                            for constraint in ci:
                                for q in constraint:
                                    if q in vars:
                                        bad_track_assignments.append((q, t))

            # solve the track assignment problem:
            tracks = solve_channel(channel_graph, channel_emb, num_tracks,
                                       working_channel_tracks,
                                       missing_channel_edges,
                                       bad_track_assignments)
            if not tracks:
                print('Failed to detailed route.')
                return None

            # convert track assignment into an embedding on hardware graph
            for x in vars:
                for q in channel_emb[x]:
                    # vertex embedding:
                    track = tracks[x][q]  # track assignment
                    v = qubit_mapping[(q, track)]
                    vertex_embedding[x].append(v)
                    which_vars[v] = x

                    if constraint_quotient_emb:
                        # constraint embedding:
                        if quotient_constraints[q] and q in \
                                quotient_terminal_emb[x]:
                            # todo: handle case where x is in this location but
                            # only as routing variable
                            # (currently the code assumes that if x has a
                            # terminal that quotient location v, and there is a
                            # constraint c at that location, then x is a
                            # terminal in that constraint).
                            for c in quotient_constraints[q]:
                                if x in c:
                                    constraint_emb[c].add(v)


    # sort embedding, constraint indices:
    for x in vertex_embedding:
        vertex_embedding[x].sort()

    if constraint_emb:
        for c in constraint_emb:
            constraint_emb[c].tolist()

    return vertex_embedding, constraint_emb


def build_chimera_quotient_embedding(embedding, graph):

    '''
    Given an embedding in a hardware graph, construct a quotient embedding.
    This can be used to build a new embedding using anneal offsets.

    Args:
        embedding: dict
            emb[x] is the chain for variable x in graph

        graph: Networkx graph
            hardware graph

    Returns:
        quotient_embedding: dict{}
            quotient_embedding[x] is the chain for variable x in
            the quotient graph
    '''

    _, quotient_mapping, _ = get_chimera_quotient_graph(graph)
    embedding = {x:[quotient_mapping[v][0] for v in chain] for x, chain in
                  embedding.items()}

    return embedding
