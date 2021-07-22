import numpy as np
"""
Conjugate residual method: https://en.wikipedia.org/wiki/Conjugate_residual_method
"""
def CR(A,b, x0):
    m, n = A.shape
    assert(m == n and m == b.shape[0] and b.shape == x0.shape)
    residual = 1.0e-6
    r0 = b - A @ x0
    if np.linalg.norm(r0) < residual:
        return x0
    x = x0
    p = r0
    r = r0
    count = 0
    Ap = A @ p
    while True:
        Ar = A @ r  
        rAr = sum(r * Ar)
        alpha = rAr / sum(Ap * Ap)
        x = x + alpha * p
        r_next = r - alpha * Ap
        if np.linalg.norm(r_next) < residual:
            break
        Ar_next = A @ r_next
        beta = sum(r_next * Ar_next)/ rAr
        p = r_next + beta * p
        r = r_next
        Ap = Ar_next + beta * Ap 
        count += 1
    return x, count

A = np.array([[1,0],[0, 2]])
b = np.array([5,6]).reshape(2,1)
x0 = np.zeros((2,1))

x = np.linalg.inv(A) @ b
print(f"Real solution: {x}")
tx, numIter  = CR(A,b,x0)
print(f"T solution: {tx}")
