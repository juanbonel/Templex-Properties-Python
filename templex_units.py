#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 08:12:30 2025

@author: Juan Cruz Bonel
"""
# -*- coding: utf-8 -*-
"""
BraMAH COMPLEX
"""


''' This notebook computes templex properties (homologies, orientability chain, generatex, stripex) from a given 
generating 2-complex and its associated digraph following definitions in Bonel, J. C, Bodnarink, N., Charo, Gisela D., Letellier, C.,
Guinet, C., Saraceno, M. and Sciamarella, D.; "Templex for Lagrangian dynamics in the Southwestern Atlantic". Chaos: An Interdisciplinary 
Journal of Nonlinear Science, 35, 103137 (2025) (https://doi.org/10.1063/5.0255611). Also this code computes the templex units, algorithm associated
with Mosto C., Charó, G. D., Letellier, C., Sciamarella D.: "Templex-based dynamical units for a taxonomy of chaos". Chaos 34, 113111 (2024).(https://doi.org/10.1063/5.0233160)'''

import networkx as nx
import copy
import numpy as np
import olll
from pyvis.network import Network
from collections import defaultdict
from collections import OrderedDict
import itertools
from itertools import chain
from itertools import permutations 
from sympy import Matrix


  

def templex(Poligons, Inputflow):
    #       Templexes from Bonel et al., 2025

    #     Launch computations involving  generating cell complex
    
    # =============================================================================
    #     HOMOLOGY GROUPS
    # =============================================================================
    
    # These comands follow [Sciamarella & Mindlin, 1999 & 20]. 

    S0=list(set(sorted(list(np.concatenate(Poligons).flat))))

    # I store in S2 the polygons with the last vertex equal to the first one.
    S2 = []
    for lv in Poligons:
        S2.append(lv + [lv[0]])

    # I place in S1 the 1-cells of the polygons in canonical order without repeats.
    S1=list()
    for i in range(len(S2)):
        for j in range(1,len(S2[i])):
            par=sorted([S2[i][j-1],S2[i][j]])
            if par not in S1:
                S1.append(par)
    S1 = sorted(S1)

    # I assemble borders1 by putting for each 1-cell of S1 a 1 and a -1 in its vertices such that the 
    # #orientation of the 1-cells obeys the canonical order of the vertex labels.
    borders1=np.repeat(0,len(S1)*len(S0)).reshape(len(S1),len(S0))
    for i in range(len(S1)):
        row1=[0]*(len(S0))
        for j in range(len(S0)):
            if S0[j]==S1[i][0]:
                row1[S0.index(S1[i][0])]=-1
        for j in range(len(S0)):
            if S0[j]==S1[i][1]:
                row1[S0.index(S1[i][1])]=1
        borders1[i,]=row1

    # I assemble borders2 as follows: I count the vertices of each 2-cell, take each 1-cell  
    # from its edge and I check how it is oriented (if it is in canonical order I put + 1 and if not - 1).
    borders2=np.repeat(0,len(S2)*len(S1)).reshape(len(S2),len(S1))
    for i in range(len(S2)):
        row0=[0]*(len(S1))
        for j in range(1,len(S2[i])):
            par=[S2[i][j-1],S2[i][j]]
            pospar=S1.index(sorted(par))
            if par==sorted(par):
                sgpar=1
            else:
                sgpar=-1
            row0[pospar]=sgpar
            borders2[i,]=row0  


    # Calculation of null cycles and torsion generators
    zeroone=[]
    torsiongroup=str
    highestmodulus=[]
    nullcycles =[]
    torsiongenerator = []

    sumborders2 = np.sum(borders2, axis=0)
    highestmodulus=max(sumborders2)
    
    if highestmodulus>1:
        for n in range(len(S1)):
            discriminant = sumborders2[n]
            simplices = S1[n]
            if discriminant!=0:
                zeroone.append(simplices)
                if discriminant>0:
                    nullcycles.append(str(discriminant)+str(simplices))
                    torsiongenerator.append(str(simplices))
                else:
                    nullcycles.append(str(discriminant)+str(simplices))
                    torsiongenerator.append(str(np.flip(simplices)))
        

    # =============================================================================
        #     Calculation of Bk: groups as the LI rows of the transpose matrices
        #     Calculation of Zk: groups as the kernels of the transpose matrices
    # =============================================================================


    tborders2=np.transpose(borders2)
    v2 = borders2.shape[0]
    vnull2 = [0]*v2

    tborders1=np.transpose(borders1)
    v1=borders2.shape[1]
    vnull1 = [0]*v1


    v0 = len(borders1[1])
    vnull0 = [0]*v0

    tborders2u=Matrix(tborders2)
    Z2= tborders2u.nullspace()
    if len(Z2)==0:
        Z2.append(vnull2)

    #H2
    H2=Z2
    B2=vnull2

    # =============================================================================
    #     Z1, B1, H1, X1
    # =============================================================================
    # B1: search for the LI rows of the transpose borders2 matrix   
    tborders1=Matrix(tborders1)
    Z1=tborders1.nullspace()

    if len(Z1)==0:
        Z1.append(vnull1)

    # B1 setting to be able to use Matrix()
    B1i=np.repeat(0,v1*v2).reshape(v2,v1)
    for i in range(v2):
        for j in range(v1):
            B1i[i,j]=borders2[i,j]
    B1iok=Matrix(B1i)

    # olll.reduction finds the LI rows and stores them in B1
    B1=[]
    for s in range(v2):
        for t in range(v1):
            if abs(borders2[s][t])>1:
                B1 = olll.reduction(B1i)
    #rref makes the Gaussian elimination
    if len(B1)==0:
        B1=B1iok.rref()
        B1=np.array(B1[0])


    B1=np.stack(B1).astype(None)
    B1=np.array(B1)

    # Search H1
    b1 = len(B1)
    z1 = len(Z1)
    X1=[]
    Z1=reversed(Z1)
    Z1 = np.stack(list(reversed(list(Z1)))).astype(None)
    Z1=Z1[:,:,0]

    #X1: least squares solution of Z1 X1= B1
    for k in range(0,b1):
        X1.append(np.linalg.lstsq(np.transpose(Z1),B1[k],rcond=float(0))[0])
    X1= np.round(X1)


    #Array must be floats for rref funcion
    XR1bis=rref(X1)

    # put XR1bis in an XR1 array
    XR1=np.repeat(0,b1*z1).reshape(b1,z1)
    for k in range(0,b1):
        vector=XR1bis[k][::]
        for m in range(0,z1):
            XR1[k,m]=vector[m]
    XR1 = XR1.astype(int)
    X1 = X1.astype(int)

    # Factor1: I store the rows that are equal or multiples between XR1 and X1.
    FACTOR1=np.repeat(0,b1*b1).reshape(b1,b1)
    factor1=[0]*b1
    for i in range (0,b1):
        for j in range (0,b1):
            for m in range(0,z1):
                if X1[i][m]!=0:
                    k=XR1[i][m]/X1[j][m]
                if all(XR1[i]==k * X1[j]):
                    FACTOR1[j,i]=k
                    factor1[j]=k

    H1 = []
    H1Comp = []
    Nceros1=[] # I count the number of zeros per row of XR1.
    for i in range(0,b1):
        counter=0
        for j in range(0,z1):
            if XR1[i,j]==0:
                counter=counter+1
        Nceros1.append(counter)

    for i in range(0,b1):
        if Nceros1[i]==(z1-1) and abs(factor1[i])<=1:
            for j in range (0,z1):
                if XR1[i,j]==1 or XR1[i,j]==-1:
                    H1Comp.append(Z1[j])
                
    for i in range (0,b1):
        if Nceros1[i]<z1-1 and abs(factor1[i]<=1):
            for j in range(0,z1):
                if XR1[i,j]!=0:
                    H1Comp.append(Z1[j])
                    break

    H1Comp=np.stack(H1Comp).astype(None)

    vector=[]
    for i in range(0,z1):
        counter=0
        for j in range(0,len(H1Comp)):
            if all(Z1[i]==H1Comp[j]):
                counter=counter+1
        if counter==0:
            H1.append(Z1[i])


    # Detect whether H1 has a string or not

    for i in range(0,b1):
        if factor1[i]==int(factor1[i]) and abs(factor1[i]>1):
            for j in range(0,z1):
                suma=sum(X1[i,j])*Z1[j,:]
                H1.append("tal que"+str(suma)+"=0")
        
    # =============================================================================
    #     Z0, B0, H0, X0
    # =============================================================================
    Z0=np.identity(v0)

    borders1bis = borders1.astype(float)
    B0bis=rref(borders1bis)

    # B0bis accommodation in B02
    B02=np.repeat(0,v1*v0).reshape(v1,v0)
    for k in range(0,v1):
        vector=B0bis[k][::]
        for m in range(0,v0):
            B02[k,m]=vector[m]

    B0=[]
    # I add to B0 only those that are not null vectors.
    for i in range(0,len(B02)):
        if all(B02[i]==0):
            pass
        else:
            B0.append(B02[i])
        
    b0 = len(B0)
    z0 = len(Z0)
    X0=[] #X0 least-squares solution of Z0 X0= B0
    for i in range(len(B0)):
        X0.append(np.linalg.lstsq(np.transpose(Z0),B0[i],rcond=float(0))[0])
    XR0= np.round(X0)

    H0=[]
    H0Comp=[]
    Nceros0=[] #count the number of zeros per row of XR0
    for i in range(b0):
        counter=0
        for j in range(z0):
            if XR0[i,j]==0:
                counter=counter+1
        Nceros0.append(counter)

    for i in range(b0):
        if Nceros0[i]<=z0-1:
            for j in range(z0):
                if XR0[i,j]!=0:
                    H0Comp.append(Z0[j])
                    break
            
    H0Comp=np.stack(H0Comp).astype(None)

    for i in range(z0):
        counter=0
        for j in range(len(H0Comp)):
            if all(Z0[i]==H0Comp[j]):
                counter=counter+1
        if counter==0:
            H0.append(Z0[i])

    if len(Z0)==0:
        Z0.append(vnull0)
    if len(B0)==0:
        B0.append(vnull0)
    if len(H0)==0:
        H0.append(vnull0)
    if len(Z1)==0:
        Z1.append(vnull1)
    if len(B1)==0:
        B1.append(vnull1)
    if len(H1)==0:
        H1.append(vnull1)    
    if len(Z2)==0:
        Z2.append(vnull2)
    if len(B2)==0:
        B2.append(vnull2)
    if len(H2)==0:
        H2.append(vnull2)

    if H0==vnull0:
        connectedcomponents=0
    else:
        connectedcomponents=len(H0)

    for i in range(len(H1)):
        if (H1[i] - vnull1).all():
            loops=0
        else:
            loops=len(H1)
            
    for i in range(len(H2)):
        if H2[i]==vnull2:
            cavities=0
        else:
            cavities=len(H2)            


    if connectedcomponents==0:
        print("H0~vacío")
    else:
        print("H0~Z"+str(connectedcomponents))

    if highestmodulus<=1:
        print("No orientability chains.")
    else:
        print("Orientability chain given by 01 =" +str(nullcycles))

    print("NUMBER OF 0-GENERATORS: "+ str(connectedcomponents))
    print("NUMBER OF 1-GENERATORS: "+ str(loops))
    print("NUMBER OF 2-GENERATORS: "+ str(cavities))

    # =============================================================================
    #     H1, H2 and torsions.
    # =============================================================================

    # I create armo Loops[] to store them separately and in the order in which they can be plotted
    Loops=[0]*len(H1)
    Cavities=[0]*len(H2)

    for i in range(len(H1)):
        Loops[i]=[]
    for i in range(len(H2)):
        Cavities[i]=[]

    loops=[]
    oneloop=[]

    for m in range(len(H1)):
        if loops!=0:
            escribir=""
            generator1=""
            for n in range(len(H1[0])):
                discriminant=H1[m][n]
                simplices=S1[n]
                if discriminant!=0:
                    Loops[m]=simplices
                if discriminant==0:
                    pass
                if discriminant>0:
                    oneloop.append(simplices)
                    generator1=str(generator1)+"+"+str(simplices)
                if discriminant<0:
                    oneloop.append([simplices[1],simplices[0]])
                    generator1=str(generator1)+"-"+str(simplices)
                if n==len(H1[0])-1 and m<=len(H1[0])-1:
                    loops.append(oneloop)
                escribir = str(generator1)
            print("1-GENERATOR:",escribir)


    write=""
    generator2=""
    for m in range(len(H2)):
        if cavities==0:
            print("H2 ~ vacío")
        if cavities!=0:
            for n in range(len(H2[0])):
                discriminant=H2[m][n]
                simplices=S2[n]
                if discriminant==0:
                    pass
                if discriminant>0:
                    generator2=str(generator2)+"+"+str(simplices)
                if discriminant<0:
                    generator2=str(generator2)+"-"+str(simplices)
        if loops != 0:
            write = []

    writegenerator2=str(write)+str(generator2)
    print("2-GENERATOR:"+writegenerator2)


    if torsiongroup!=str:
        print("Torsion generator="+str(writegenerator2))
        print("Torsion coefficient=", str(highestmodulus))
        
        
    # =============================================================================
    #     JOINING LOCUS

    # To calculate the joining locus, we must find at least three two-cells 
    # that share an edge. This is done with the transpose of the matrix of 
    # borders, asking that the rows in modulo add up to more than three or three.
    # =============================================================================

    joinings=0
    joiningrows=[]
    joining1cells=[]


    trasp2=np.transpose(borders2)
    for nr in range(len(trasp2[:,])):
        if sum(abs(trasp2[nr,]))>=3:
            joinings=joinings +1
            joiningrows.append(trasp2[nr,])
            joining1cells.append(S1[nr])
            

    # I define the function spchain: given the 1-cell number of the 
    # shared edge, it returns the two cells it has attached.

    def spchain(i):
        vector=[]
        for j in range(len(joiningrows[i])):
            if joiningrows[i][j]!=0:
                vector.append(j)
        return vector


    joining2chainslong = ""
    joining2chains = []
    joining2cells = [] 
    joining2list = []

    for ns in range(joinings):
        for nc in range(len(spchain(ns))):
            added=S2[spchain(ns)[nc]]
            multiply=joiningrows[ns][spchain(ns)[nc]]
            joining2chainslong=joining2chainslong+"("+str(multiply)+")"+"*"+str(added)+","
            sumo=[e1 + e2 for e1, e2 in zip(spchain(ns),[1]*len(spchain(ns)))] 
            joining2cells.append(sumo)
            joining2list.append(spchain(ns))
            joining2chains.append((joiningrows[ns][spchain(ns)[nc]])*([spchain(ns)[nc]]))


    joining2cells=list(joining2cells for joining2cells,_ in itertools.groupby(joining2cells))

    if len(joining2list)>1:
        joining2list = list(np.concatenate(joining2list).flat)
        joining2list=[y+1 for y in joining2list]
        joining2list=set(joining2list)
    else:
        joining2list=[]

    print("Number of joining 1-cells = "+str(joinings))
    print("Joining 1-cells = "+str(joining1cells))
    print("Joining 2-cells = "+str(joining2cells))
    print("Joining 2-chain = "+str(joining2chainslong))


    # =============================================================================
    #     LAUNCH TEMPLEX COMPUTATIONS (involving complex and digraph)
    #     These commands follow [Charó et al, 2023] Sciamarella, D. & Charó, D. G. 
    #     (to be published). New elements for a theory of chaos topology, chapter 
    #     contributed to the Springer volume: Topological Methods for Delay and 
    #     Ordinary Differential Equations. 
    # =============================================================================

    NrIF = len(Inputflow) # number of edges in graph

    #Nr2cells = len(S2) #number of 2-cells

    T0 = []
    T1 = []
    k0 = 0
    k1 = 0

    #T0 joins the initial nodes of the connections and T1 the final nodes. 

    while k0<NrIF:
        T0.append(Inputflow[k0][0])
        k0=k0+1
    while k1<NrIF:
        T1.append(Inputflow[k1][1])
        k1=k1+1
        

    #F2 is a T0xT1 matrix that sets a 1 if there is a connection and zeros otherwise. 
    F2=np.repeat(0,len(Inputflow)*len(Inputflow)).reshape(len(Inputflow),len(Inputflow))
    for i in range(len(Inputflow)):
        a=T0[i]-1
        b=T1[i]-1
        F2[a,b]=1


    #SubGraph gives in list format the part of the network describing 
    #the flow between two cells that belong to the joining line. 
    #It is necessary to discriminate between the two cells that are ingoing 
    # and outgoing with respect to the joining line. 


    subgraph=[]

    for i in range(len(joining2cells)):
        for e in range(len(Inputflow)):
            for b in range(len(joining2cells[i])):
                for a in range(len(joining2cells[i])):
                    if [joining2cells[i][a],joining2cells[i][b]]==Inputflow[e]:
                        if sum(F2[:,joining2cells[i][b]-1])>1:
                            subgraph.append([joining2cells[i][a],joining2cells[i][b]])

    Outgoing2Cells=[]
    Ingoing2Cells=[]
    for i in range(len(subgraph)):
        if subgraph[i][0] not in Ingoing2Cells:
            Ingoing2Cells.append(subgraph[i][0])
        if subgraph[i][1] not in Outgoing2Cells:
            Outgoing2Cells.append(subgraph[i][1])
            

    # Create Directed Graph
    G=nx.DiGraph()

    nodes=list(range(1,max(max(Inputflow))+1))

    # Add a list of nodes:
    G.add_nodes_from(nodes)

    # Add a list of edges:
    G.add_edges_from(Inputflow)

    #Return a list of cycles described as a list o nodes
    subcomplex=list(nx.simple_cycles(G))


    truejoining2cells = []  
    truejoining1cells = []  
    inoutmatrix = nx.to_pandas_adjacency(G)
    indegree = inoutmatrix.sum(axis=0)
    for j in range(len(joining2cells)):
        for i in range(len(joining2cells[j])):
            if indegree[joining2cells[j][i]]>=2.0:
                truejoining2cells.append(joining2cells[j])
                truejoining1cells.append(joining1cells[j])
                break

    # The sign of the 2-cells alla Christophe is used to reorient the 1-cells
    # of the joining line in such a way that it is clear whether or not there 
    # is or is not a 0-cell that splits the flow, i.e. a critical point.
    SC=[0]*len(subcomplex)

    for i in range(len(subcomplex)):
        scl=[]
        for j in range(len(subcomplex[i])):
            scl.append(S2[subcomplex[i][j]-1][:-1])
        SC[i]=scl

    out=0

    # Out gives the outgoing corresponding to that truejoining2cell.
    sign=[0]*len(truejoining2cells)
    for i in range(len(truejoining2cells)):
        for j in range(len(truejoining2cells[i])):
            if truejoining2cells[i][j] in Outgoing2Cells:
                out=truejoining2cells[i][j]
        sign[i]= tborders2[S1.index(truejoining1cells[i]),out-1]

            
    ReorientedJoining1chain=""
    for i in range(len(sign)):
        multiply=sign[i]
        added=joining1cells[i]
        ReorientedJoining1chain=ReorientedJoining1chain+"("+str(multiply)+")"+"*"+str(added)

    Pos=[]
    Neg=[]
    for rjc in range(len(truejoining1cells)):
        if sign[rjc]==1:
            Pos.append(truejoining1cells[rjc])
        if sign[rjc]==-1:
            Neg.append(truejoining1cells[rjc])    


    for i in range(len(truejoining1cells)):
        vnull1[S1.index(truejoining1cells[i])]=sign[i]
    VV=vnull1

    CritPoint=0
    prodvect=np.dot(tborders1,VV)
    if any(prodvect==2):
        CritPoint=2


    print("Joining subgraph = ", subgraph)
    print("Ingoing 2-cells in subgraph = ", Ingoing2Cells)
    print("Outgoing 2-cells in subgraph = ", Outgoing2Cells)
    print("Sub-complexes in terms of indexed 2-cells = ", subcomplex)
    print("Sub-complexes in terms of indexed 0-cells for further analysis= ", SC)


    print("*********************************************************************************************\
    Rule: orientation cannot be propagated through the merging line. *********************************************************************************************\
    ")

    print("Reoriented Joining Locus = ", ReorientedJoining1chain)
    print("Splitting 0-cell = ", CritPoint)
    print("Positive segment = ", Pos)
    print("Negative segment = ", Neg)


    # Plot
    g = Network(notebook=True,directed =True)
    for i in range(len(nodes)):
        g.add_node(nodes[i],label=str(nodes[i]))

    for i in range(len(Inputflow)):
        g.add_edge(Inputflow[i][0],Inputflow[i][1])

    g.show('Graph.html')


    # =============================================================================
    #    STRIPEXES
    # =============================================================================

    # These comands follow [Charo et al. 2022]. 

    #OrderMax contains the intersection between SubComplexes and joining2list
    OrderMax=0
    orderlist=[]
    for i in range(len(subcomplex)):
        orden=len([val for val in subcomplex[i] if val in joining2list])
        orderlist.append(int((orden/2)))
        if orden>OrderMax:
            OrderMax = int(len([val for val in subcomplex[i] if val in joining2list])/2)


    # I have the maximum order, a list with the orders and all the subcomplexes. 
    # I'm going to order them by putting first the outgoing cell returns all the 
    # positions of outgoing2cells preceded by ingoing2cells.

    def Posfirstout(_list):
        positions=[]
        for i in range(len(Outgoing2Cells)):
            if Outgoing2Cells[i] in _list:
                valor=_list.index(Outgoing2Cells[i])-1
                if _list[valor] in Ingoing2Cells:
                    positions.append(_list.index(Outgoing2Cells[i]))
        return positions

    #I keep all cycles in GENERATEX for later deletion of duplicates.
    
    GENERATEX=[]
    if len(Outgoing2Cells)!=0:
        #I order so that all cycles start with one outgoing and I order the order of the second one.
        for i in range(len(subcomplex)):
            firstout=Posfirstout(subcomplex[i])[0]
            subcomplex[i]=list(np.roll(subcomplex[i],-firstout))
            if orderlist[i]==1:
                GENERATEX.append(subcomplex[i])
            if orderlist[i]>=2:
                a=Posfirstout(subcomplex[i])
                b=[firstout]*len(Posfirstout(subcomplex[i]))
                outgoings=[e1 - e2 for e1, e2 in zip(a,b)]
                outgoings.append(len(subcomplex[i]))
                newsplitlist=list()
                for j in range(0,len(outgoings)-1):
                    newsplitlist.append(subcomplex[i][outgoings[j] : outgoings[j+1]])
                    GENERATEX.append(subcomplex[i][outgoings[j] : outgoings[j+1]])
                subcomplex[i]=newsplitlist



        GeneratexSet = GENERATEX

        StripexSet = []


        for i in range(len(orderlist)):
            if orderlist[i]>1:
                for j in range(len(subcomplex[i])):
                    if subcomplex[i][j] not in StripexSet:
                        StripexSet.append(subcomplex[i][j])
            else:
                if subcomplex[i] not in StripexSet:
                    StripexSet.append(subcomplex[i])

        # I add the first value (outgoing cell) to the end
        for i in range(len(GeneratexSet)):
            for j in range(len(Outgoing2Cells)):
                if [GeneratexSet[i][-1],Outgoing2Cells[j]] in Inputflow:
                    GeneratexSet[i].append(Outgoing2Cells[j])



        print("Generatex set = ", GeneratexSet)
        print("Stripex set = ", StripexSet)

        # =============================================================================
        #    REDUCED DIGRAPH


        # Main nodes: splittings and joinings 2-cells (the latter includes outgoings).
        # It cannot happen that a main node is splitting and joining at the same time. If this
        # happens, I will merge the original network.

        # =============================================================================

        # Calculation of how many nodes to merge
        counter=0
        mainnodes=list()
        for i in range(1,len(Poligons)+1):
            if G.out_degree(i) > 1 and G.in_degree(i) > 1:
                counter=counter+1
        nodes_to_be_merged=[]*counter
        edges_a_fusionar=[]*counter

        
        # I store in nodes_to_be_merged the nodes that come out of the splitting cell.
        # This means that I move the splitting cell to the next (merged) cell.
        for i in range(1,len(Poligons)+1):
            if G.out_degree(i) > 1 and G.in_degree(i) > 1:
                nodes=[]
                edges=[]
                for j in range(len(Inputflow)):
                    if Inputflow[j][0]==i:
                        nodes.append(Inputflow[j][1])
                        edges.append(Inputflow[j])
                nodes_to_be_merged.append(nodes)


        # I contract the nodes to be merged and replace them in GeneratexSet by the first node
        # that I merged which will be the label of the (merged) barra node.
        for i in range(counter):
            G=nx.contracted_nodes(G, nodes_to_be_merged[i][0], nodes_to_be_merged[i][1])
            #G=nx.contracted_edge(G, edges[i][1])
            mainnodes.append(nodes_to_be_merged[i][0])
            for j in range(len(GeneratexSet)):
                for m in range(len(GeneratexSet[j])):
                    if GeneratexSet[j][m]==nodes_to_be_merged[i][1]:
                        GeneratexSet[j][m]=nodes_to_be_merged[i][0]
                        
        for i in G.nodes:
            if G.out_degree(i) > 1 or G.in_degree(i) > 1:
                mainnodes.append(i)


        # I split the GeneratexSet for each main node.

        def split_at_values(lst, values):
            _index = [i for i, x in enumerate(lst) if x in values]
            for start, end in zip([0, *_index], [*_index, len(lst)]):
                yield lst[start:end+1]
                # Note: remove +1 for separator to only appear in right side slice

        split_generatex_long = list(chain.from_iterable(split_at_values(sublst, mainnodes) for sublst in GeneratexSet))

        # I remove empty lists or outgoings from the output list.

        split_generatex=[x for x in split_generatex_long if len(x)>1]

        path=["0"]*len(split_generatex)


        for i in range(len(split_generatex)):
            if split_generatex[i][0]==split_generatex[i][-1]:
        # Within the first ones, only those with 4 or more cells are contemplated, so that you can
        # triangular (the first and the last are equal). The first and the last 3 cells are saved.        
                path[i]=[split_generatex[i][0],split_generatex[i][-3],split_generatex[i][-2],split_generatex[i][-1]]
        #Within the second case, 3 cells are kept for those cases where the first 
        # is not outgoing and the last is outgoing, and only the first and last for the rest of the cases.

            else:
                if split_generatex[i][-1] in Outgoing2Cells and split_generatex[i][0] not in Outgoing2Cells:
                    path[i]=[split_generatex[i][0],split_generatex[i][-2], split_generatex[i][-1]]
                else:
                    path[i]=[split_generatex[i][0], split_generatex[i][-1]]


        # Remove duplicates

        def get_unique_list(seq):
            seen = []
            return [x for x in seq if x not in seen and not seen.append(x)]

        nod=get_unique_list(path)
        nodesreduced=list(set(list(itertools.chain(*nod))))


        # Graphic the reduced digraph

        rg = Network(notebook=True,directed =True)
        for i in range(len(nodesreduced)):
            rg.add_node(int(nodesreduced[i]),label=str(nodesreduced[i]))

        Graph=[]
        for i in range(len(nod)):
            rg.add_edge(int(nod[i][0]),int(nod[i][1]))
            Graph.append([int(nod[i][0]),int(nod[i][1])])
            if len(nod[i])>2:
                rg.add_edge(int(nod[i][1]),int(nod[i][2]))
                Graph.append([int(nod[i][1]),int(nod[i][2])])
            if len(nod[i])==4:
                rg.add_edge(int(nod[i][2]),int(nod[i][3]))
                Graph.append([int(nod[i][2]),int(nod[i][3])])


        rg.show('Reduceddigraph.html')

        Inputflow=Graph

        # =============================================================================
        #    TEMPLEX UNITS 

        # These comands follow [Mosto et al. 2024].

        # Bonds are searched as the edges going from an OUTGOING cell (Indregree > 1) 
        # to a SPLITTING cell (Outdegree > 1). SPLITTING cell (Outdegree > 1)
        # There is one O-unit per bond
        # The s-units are the units that do not contain bond and contain only one of 
        # the edges of SubComplexes.
        # =============================================================================
    
        # I define a network and an adjacency matrix 
        G=nx.DiGraph(Inputflow)
        inoutmatrix = nx.to_pandas_adjacency(G)
        indegree = inoutmatrix.sum(axis=0)
        outdegree = inoutmatrix.sum(axis=1)

        # Looking for outgoings and splittings
        splitting=[]
        outgoing=[]
        for i in range(len(outdegree)):
            if outdegree.iloc[i]>1:
                splitting.append(list(G.nodes)[i])
            if indegree.iloc[i]>1:
                outgoing.append(list(G.nodes)[i])

        # Of the possible combinations of OUT--> SPLITTING, I will stick with the existing ones
        Bonds=[]
        unique_combinations = []
        for i in range(len(outgoing)):
            for j in range(len(splitting)):
                unique_combinations.append((outgoing[i], splitting[j]))
        for i in range(len(unique_combinations)):
            if list(unique_combinations[i]) in Inputflow:
                Bonds.append(list(unique_combinations[i]))

        print('Bond:',Bonds)

        # I am looking for direct directed cycles
        C=list(nx.simple_cycles(G))

        # I order SubComplexes so that the shortest cycle comes first
        C = sorted(C, key=len)

        for i in range(len(C)):
            for j in range(len(C[i])):
                if [C[i][j]]==outgoing:
                    C[i]=C[i][j:]+C[i][:j]

        # I assign O-units
        # np.unique() removes repeats from a list
        # scroll C:
        # if it contains the bond and was not yet added to ounits:
        # if more than one bond exists and bonds flatten does not belong to C.
        # subset of all bonds

        # This cycle is useful for NON-degenerate complexes:
        Ounits=[]
        for i in range(len(Bonds)):
                for j in range(len(C)):    
                        Bondsf=[item for _list in Bonds[:(i+2)] for item in _list]
                        if set(np.unique(Bonds[i])).issubset(C[j]) and C[j] not in Ounits:
                                if len(Bonds) > 1 and not set(np.unique(Bondsf)).issubset(C[j]):
                                        Ounits.append(C[j])
                                if len(Bonds) == 1:
                                        Ounits.append(C[j])
                                        break
        if len(Ounits)!=0:
                Sunits=['0']*(len(C)-len(Ounits))

        # For degenerate cases: at the end I take the O as 2.
        if len(Ounits)==0:
                for i in range(len(Bonds)):
                        for j in range(len(C)):     
                                Bondsf=[item for _list in Bonds[:(i+2)] for item in _listt]
                                if set(np.unique(Bonds[i])).issubset(C[j]) and  C[j] not in Ounits:
                                        Ounits.append(C[j])
                                        break
                                if len(Ounits)==1:
                                        break

        # REDUCED DIGRAPH WITH UNITS
        rg_units = Network(notebook=True,directed =True)
        
        for i in range(len(nodesreduced)):
            rg_units.add_node(int(nodesreduced[i]),label=str(nodesreduced[i]))
        
        Graphr=[]

       # Draw Ounits 
        for i in range(len(Ounits)):
            rg_units.add_edge(Ounits[i][-1],Ounits[i][0], color='blue',width=8,alpha=0.5)
            res = list(zip(Ounits[i], Ounits[i][1:]))
            for j in range(len(res)):
                rg_units.add_edge(res[j][0], res[j][1],color= "blue",width=8,alpha=0.5)
            

        Sunits=['0']*(len(C)-len(Ounits))

        # I'm going to use the SubComplex list:
        # I take out the Ounits from SubComplexes.
        for i in range(len(Ounits)):
            C.remove(Ounits[i])

        # ASSEMBLY
        # Sack of SubComplexes the edges of the o-units
        mounting=[]
        for i in range(len(Ounits)):
            for first, second in zip(Ounits[i], Ounits[i][1:]):
                mounting.append([first, second])
            mounting.append([Ounits[i][-1],Ounits[i][0]])

        for i in range(len(C)):
            edges=[]
            newunit=[]
            for first, second in zip(C[i], C[i][1:]):
                edges.append([first, second])
            edges.append([C[i][-1],C[i][0]])
            for j in range(len(edges)):
                if edges[j] not in mounting:
                    mounting.append(edges[j])
                    newunit.append(edges[j])
            Sunits[i] = list(OrderedDict.fromkeys([item for row in newunit for item in row]))


        # I eliminate empty lists (only happens when the O is degenerate).
        for i in range(len(Sunits)):
            if len(Sunits[i])==0:
                Sunits.remove(Sunits[i])

        edge = []
        Sunitsfinal=copy.copy(Sunits)
        for i in range(len(Sunits)):
            for j in range(len(Sunits[i])-1):
                edge=[Sunits[i][j], Sunits[i][j+1]]

                if edge in Bonds:
                    Sunitsfinal[i]=[[Sunits[i][:j+1]],[Sunits[i][j+1:]]]

        
        # Draw Sunits                
     #   for i in range(len(Sunitsfinal)):
      #      res = list(zip(Sunitsfinal[i], Sunitsfinal[i][1:]))
       #     for j in range(len(res)):
                #rg_units.add_edge(res[j][0], res[j][1],color= 'red')

        
        #rg_units.show('Graph_units.html')  
              
                
        print('O-units:',Ounits)
        print('S-units:',Sunitsfinal)


# Gaussian elimination function
def rref(A):
  #copy matrix A before transformations begin in order to preserve matrix A
  a = [[A[i][j] for j in range(len(A[i]))] for i in range(len(A))]
  #initiate pivot counter
  j = 0
  #the forward phase:
  #gaussian-elimination
  for i in range(len(a)):
      #if leading term is 0, swap the row with the next non-zero row.
      #if no non-zero leading term in rest of column, move pivot counter to next column and try again.
      while a[i][j] == 0:
        for I in range(len(a)-i):
          if a[i+I][j] != 0:
            rowSwap = a[i]
            a[i] = a[i+I]
            a[i+I] = rowSwap
            break
        #if found nonzero leading term in loop, quit and move to next part
        if a[i][j] != 0:
          break
        # if arrive at last column, quit
        if j == len(a[0])-1:
          break
        j += 1
      #if leading term is not 0, turn it into 1, then turn every number below it into 0
      if a[i][j] != 0:
        #let the leading coefficient of the row remain constant as b
        b = a[i][j]
        #divide every entry in entire row by leading coefficient to create leading 1
        for h in range(len(a[0])):   
          a[i][h] = a[i][h]/b
          #round -0.0 to 0, 1.0 to 1, 2.0 to 2, etc.
          if type(a[i][h]) is float:
            if a[i][h] == int(a[i][h]):
              a[i][h] = int(a[i][h])
        #subtract leading coefficient of row times row with leading 1 from entire row for each row below row with leading 1
        for h in range(len(a)):
          if h > i:
            b = a[h][j]
            for k in range(len(a[0])):
              a[h][k] = a[h][k] - b*a[i][k]
              #round -0.0 to 0, 1.0 to 1, 2.0 to 2, etc.
              if type(a[h][k]) is float:
                if a[h][k] == int(a[h][k]):
                  a[h][k] = int(a[h][k])
      #if pivot counter is at the last column, quit. otherwise, continue finding pivots.
      if j == len(a[0])-1:
        break
      else:
        j += 1
  #the backward phase:
  #back-substitution
  while i >= 0:
      #starting on the last row, find the leading 1, and turn every term above it into 0
      for J in range(j+1):
        if a[i][J] == 1:
          j = J
          #subtract the entry above pivot point times the entire pivot row from every row
          for h in range(len(a)):
            if h < i:
              b = a[i-h-1][j]
              for k in range(len(a[0])):
                a[i-h-1][k] = a[i-h-1][k] - b*a[i][k]
                #round -0.0 to 0, 1.0 to 1, 2.0 to 2, etc.
                if type(a[i-h-1][k]) is float:
                  if a[i-h-1][k] == int(a[i-h-1][k]):
                    a[i-h-1][k] = int(a[i-h-1][k])
          break
      i -= 1
  return a

