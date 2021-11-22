import numpy as np
def CG(A,b, x0):
    m, n = A.shape
    assert(m == n and m == b.shape[0] and b.shape == x0.shape)
    residual = 1.0e-6
    r0 = b - A @ x0
    print(f"r0: {r0}")

    if np.linalg.norm(r0) < residual:
        return x0
    x = x0
    p = r0
    r = r0
    count = 0
    epsNew = sum(r * r)
    while np.linalg.norm(r) > residual:
        Ap = A @ p 
        alpha = epsNew / sum(p * Ap)
        x = x + alpha * p
        r_next = r - alpha * Ap
        epsOld = epsNew
        epsNew = sum(r_next * r_next)
        beta = epsNew / epsOld 
        p = r_next + beta * p
        r = r_next
        count += 1
    return x, count

A = np.array([[1,0],[0, 2]])
b = np.array([5,6]).reshape(2,1)
x0 = np.zeros((2,1))

x = np.linalg.inv(A) @ b
print(f"Real solution:\n {x}")
tx, numIter  = CG(A,b,x0)
print(f"T solution: \n {tx}")
