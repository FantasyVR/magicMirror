import sys 
sys.path.append('..')
import numpy as np
from obj_reader.readObj import Objfile

from scipy.sparse import coo_matrix
import taichi as ti
ti.init(arch=ti.cpu)

boundaryPoints = []

def minA(A,val):
    A = A.tolil()
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            A[i,j] = min(A[i,j],val)
    A = A.tocoo()
    return A

def equOne(A):
    B = A.tolil()
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            B[i,j] = 1 if A[i,j] == 1 else 0
    return B

# Reference: https://mathproblems123.wordpress.com/2015/04/21/identifying-edges-and-boundary-points-2d-mesh-matlab/
def findBoundaryPoints(triangles, NV):
    NF = triangles.shape[0]
    # print(f"Number of faces: {NF}")
    # print(f"Triangles:\n {triangles}, first column: {triangles[:,0]}, second column: {triangles[:,1]}")
    A  = coo_matrix((np.full(NF,1),(triangles[:,0],triangles[:,1])),shape=(NV,NV))
    A += coo_matrix((np.full(NF,1),(triangles[:,1],triangles[:,2])),shape=(NV,NV))
    A += coo_matrix((np.full(NF,1),(triangles[:,2],triangles[:,0])),shape=(NV,NV))
    A = minA(A,1)
    A = np.transpose(A) + A 
    A = minA(A,1)
    # print(f"A: {A.toarray()}")
    # print(f"A^2: {(A @ A).toarray()}")
    B = (A @ A).multiply(A)
    # print(f"A @ A * A: {B.toar    ray()}")
    B = B.tolil()
    for i in range(NV):
        for j in range(NV):
            B[i,j] = 1 if B[i,j] == 1 else 0
    # Upper triangle of B is the boundary edges
    print(f"B: {B.toarray()}")

    for i in range(NV):
        result = 0
        for j in range(NV):
            result += B[i,j]
        if result > 0:
            boundaryPoints.append(i)
    # print(f"boundary point: {boundaryPoints}")

def findEdge(edges, edge):
    return {k: edges[k] for k in edges if k == edge or (k[1],k[0]) ==edge }

def findBoudaryEdge(triangles):
    edges = dict()
    for t in triangles:
        t.sort()
        if len(findEdge(edges, (t[0],t[1]))) == 0:
            edges[(t[0],t[1])] = 1
        else: 
            edges[(t[0],t[1])] += 1
        if len(findEdge(edges, (t[1],t[2]))) == 0:
            edges[(t[1],t[2])] = 1
        else:
            edges[(t[1],t[2])] += 1

        if len(findEdge(edges, (t[2],t[0]))) == 0:
            edges[(t[0],t[2])] = 1
        else:
            edges[(t[0],t[2])] += 1

    boundaryEdges = []
    for e in edges:
        if edges[e] == 1: # the boundary edge is only inside one triangle
            boundaryEdges.append(e)
            print(e)

    return boundaryEdges


def initCubeMesh(N):
    NF = 2 * N**2  # number of faces
    NV = (N + 1)**2  # number of vertices
    pos = np.zeros(shape=(NV,2),dtype=np.float64)
    f2v = np.zeros(shape=(NF,3),dtype=np.int32)

    for i in range(N+1):
        for j in range(N+1):
            k = i * (N+1) + j
            pos[k] = [i,j] 
            pos[k] = 0.05 * pos[k] + [0.25,0.25]

    for i in range(N):
        for j in range(N):
            k = (i * N + j) * 2
            a = i * (N + 1) + j
            b = a + 1
            c = a + N + 2
            d = a + N + 1
            f2v[k + 0] = [a, b, c]
            f2v[k + 1] = [c, d, a]
    return pos, f2v


def initObj(file):
    obj = Objfile()
    obj.readTxt(file)
    vertices = obj.getVertice()
    triangles = obj.getFaces()
    for vert in vertices:
        vert *= 0.4

    return vertices, triangles


if __name__ == "__main__":
    pos, f2v = initCubeMesh(2)
    # pos, f2v = initObj("../data/armadillo.txt")
    
    print("Usage: Press SPACE to show boundary points and boundary edges.")

    gui = ti.GUI("Show Boudary Points")
    pause = True
    while gui.running:
        for e in gui.get_events():
            if e.key == gui.SPACE and gui.is_pressed:
                pause = not pause
            elif e.key == gui.ESCAPE:
                gui.running = False
        gui.circles(pos, radius = 5, color=0xFF0000)
        if not pause:
            bv = findBoundaryPoints(f2v, pos.shape[0]) # boundary vertices
            be = findBoudaryEdge(f2v)
            bp = np.zeros(shape=(len(be),2))
            ep = np.zeros(shape=(len(be),2))
            for i in range(len(be)):
                bp[i] = pos[be[i][0]]
                ep[i] = pos[be[i][1]]
            gui.lines(bp,ep, radius=4,color=0x0000FF)
            
            boundary=np.zeros((len(boundaryPoints),2),dtype=np.float64)
            for i in range(len(boundaryPoints)):
                boundary[i] = pos[boundaryPoints[i]]
            gui.circles(boundary,radius=4, color=0x00FF00)
            
        gui.show()