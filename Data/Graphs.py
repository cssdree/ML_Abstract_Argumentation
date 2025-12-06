import networkx as nx
import numpy as np
import itertools

nb_neighbors = 2  #number of neighbors with which each node is joined for watts strogatz
prob_bi = 0.2  #probability that an edge is bidirectional
probs_inc = [0.05,0.1,0.15,0.2]  #probabilities that an argument is uncertain
id_counter = itertools.count(start=0)  #unique id of each graph


def WriteApx(def_args, def_atts, inc_args, inc_atts, apxpath):
    out = open(apxpath, "w")
    for arg in def_args:
        out.write(f"arg({arg}).\n")
    for arg in inc_args:
        out.write(f"?arg({arg}).\n")
    for att in def_atts:
        out.write(f"att({att[0]},{att[1]}).\n")
    for att in inc_atts:
        out.write(f"?att({att[0]},{att[1]}).\n")
    out.close()


def CreateCompletions(def_args, def_atts, inc_args, inc_atts, apxpath):
    filepath_MAX = apxpath.replace(".apx","_MAX.apx")
    filepath_MIN = apxpath.replace(".apx","_MIN.apx")
    WriteApx(def_args+inc_args, def_atts+inc_atts, [], [], filepath_MAX)
    def_atts_MIN = GetMINCompletionAttacks(def_atts, inc_args)
    WriteApx(def_args, def_atts_MIN, [], [], filepath_MIN)
    return len(def_atts_MIN)


def GetMINCompletionAttacks(def_atts, inc_args):
    if len(inc_args) == 0:
        return def_atts
    def_atts_set = set(def_atts)
    inc_args_set = set(inc_args)
    to_remove = {(arg1, arg2) for arg1, arg2 in def_atts if arg1 in inc_args_set or arg2 in inc_args_set}
    def_atts_MIN = def_atts_set-to_remove
    return list(def_atts_MIN)


class Graphs:
    def __init__(self, method, nbn, var, IAF_root):
        self.method = method
        self.nbn = nbn
        self.var = var
        self.IAF_root = IAF_root
        self.CreateSettings()
        self.CreateGraph()

    def CreateSettings(self):
        if self.method == nx.gnp_random_graph or self.method == nx.barabasi_albert_graph:
            self.settings = [self.nbn, self.var]
        elif self.method == nx.watts_strogatz_graph:
            self.settings = [self.nbn, nb_neighbors, self.var]

    def CreateGraph(self):
        if self.method == nx.gnp_random_graph:
            G = self.method(*self.settings, directed=True)
        else:
            G = self.method(*self.settings)
            G = self.MakeDirected(G)
        self.G = G

    def MakeDirected(self, G):
        diG = nx.DiGraph()
        diG.add_nodes_from(G.nodes(data=True))
        for u,v in G.edges():
            r = np.random.uniform()
            if r < prob_bi:
                diG.add_edge(u,v)
                diG.add_edge(v,u)
            elif r < (prob_bi+(1-prob_bi)/2):
                diG.add_edge(u,v)
            else:
                diG.add_edge(v,u)
        return diG

    def MakeIncomplet(self):
        """
        Transform an AF into an IAF
        """
        for prob_inc in probs_inc:
            inc_args = []
            inc_atts = []
            def_args = []
            def_atts = []
            for arg in self.G.nodes():
                if np.random.uniform() < prob_inc:
                    inc_args += [arg]
                else:
                    def_args += [arg]
            for att in self.G.edges():
                if np.random.uniform() < prob_inc:
                    inc_atts += [att]
                else:
                    def_atts += [att]
            #Check that a graph always have at least one attack between two certain args
            valid_attacks = [(u, v) for (u, v) in self.G.edges() if u in def_args and v in def_args]
            existing_valid = [(u, v) for (u, v) in def_atts if u in def_args and v in def_args]
            if len(existing_valid) == 0 and len(valid_attacks) != 0:
                def_atts.append(valid_attacks[0])
            self.Export(prob_inc, "inc", def_args, def_atts, inc_args, inc_atts)  #both arguments and attacks can be uncertains
            self.Export(prob_inc, "arg-inc", def_args, def_atts+inc_atts, inc_args, [])  #only arguments can be uncertains
            self.Export(prob_inc, "att-inc", def_args+inc_args, def_atts, [], inc_atts)  #only attacks can be uncertains

    def Export(self, p_inc, inc_type, def_args, def_atts, inc_args, inc_atts):
        filename = self.CreateFilename(p_inc, inc_type)
        WriteApx(def_args, def_atts, inc_args, inc_atts, f"{self.IAF_root}/{filename}.apx")
        len_def_atts_MIN = CreateCompletions(def_args, def_atts, inc_args, inc_atts, f"{self.IAF_root}/completions/{filename}.apx")

    def CreateFilename(self, p_inc, inc_type):
        methods = {nx.erdos_renyi_graph: "ER", nx.watts_strogatz_graph: "WS", nx.barabasi_albert_graph: "BA"}
        id = next(id_counter)
        return f"{methods[self.method]}_{self.nbn}_{self.var}_{p_inc}_{inc_type}_{id}"