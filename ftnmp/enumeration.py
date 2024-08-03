import torch as tc 
import itertools as it

def sat_clause(NODE_connect,configurat,clauses,tendencies):
    Es=[]
    for clause_id in range(len(clauses)):
        clause = clauses[clause_id]
        Ez = 1
        for node_id in range(len(clause)):
            point = clause[node_id]
            if point in NODE_connect:
                configurat_id = NODE_connect.index(point)
                Ei = (1-configurat[configurat_id]*tendencies[clause_id][node_id])/2
                Ez = Ez*Ei
        Es.append(Ez)
    E = sum(Es)
    return E,Es

def enumeration_SAT(clauses,tendencies,n_qubit,pos,spin):
    s = list(it.product(range(2), repeat=n_qubit))
    X1 = []
    for i in range(2**n_qubit):
        S = [2*si-1 for si in s[i]]
        Es = []
        for clause_id in range(len(clauses)):
            clause = clauses[clause_id]
            E=1
            for node_id in range(len(clause)):
                j = clause[node_id]
                e = (1-S[j]*tendencies[clause_id][node_id])/2
                E = e*E
            Es.append(E)        
        x1 = sum(Es)
        X1.append(x1)
    n_full = 0
    for g in X1:
        if g ==0:
            n_full=n_full+1
    X = []
    for i in range(2**n_qubit):
        S = [2*si-1 for si in s[i]]
        Es = []
        require = 0
        for pos_id in range(len(pos)):
            ei = (1-S[pos[pos_id]]*spin[pos_id])/2
            require=require+ei
        if require==0:
            for clause_id in range(len(clauses)):
                clause = clauses[clause_id]
                E=1
                for node_id in range(len(clause)):
                    j = clause[node_id]
                    e = (1-S[j]*tendencies[clause_id][node_id])/2
                    E = e*E
                Es.append(E)        
            x = sum(Es)
            X.append(x)
        n_part = 0
        for g in X:
           if g ==0:
                n_part=n_part+1
    return n_full,n_part,n_part/n_full

def marginal_enumeration(nodes,clauses,tendencies,n_qubit):
    margnals_enumeration=[]
    for tar_node in nodes:
        x,y,z = enumeration_SAT(clauses,tendencies,n_qubit,[tar_node],[-1])
        X,Y,Z = enumeration_SAT(clauses,tendencies,n_qubit,[tar_node],[1])
        mar = tc.tensor([z,Z])
        margnal_enumeration = mar/(mar[1]+mar[0])
        margnals_enumeration.append(margnal_enumeration)
    return margnals_enumeration

def enumeration_SAT_easy(clauses,tendencies,n_qubit,file):
    s = list(it.product(range(2), repeat=n_qubit))
    X1 = []
    for i in range(2**n_qubit):
        S = [2*si-1 for si in s[i]]
        Es = []
        for clause_id in range(len(clauses)):
            clause = clauses[clause_id]
            E=1
            for node_id in range(len(clause)):
                j = clause[node_id]
                e = (1-S[j]*tendencies[clause_id][node_id])/2
                E = e*E
            Es.append(E)        
        x1 = sum(Es)
        X1.append(x1)
        if x1 ==0:
            pass
    n_full = 0
    for g in X1:
        if g ==0:
            n_full=n_full+1
    return n_full