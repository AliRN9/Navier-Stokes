import numpy as np
import matplotlib.pyplot as plt

def solver(T,p_prev,u_prev, v_prev,b):
    global eps
    global A
    global N
    global delta_t
    print(eps)
    t = 0

    while t < T:
        p_new = p_prev
        norm = eps + 1
        while norm > eps:

            u_new,v_new = solve_speed(u_prev,v_prev,p_new,N,delta_t)

            b = -div(u_new,v_new)
            norm = np.linalg.norm(b) * h/delta_t
            if norm > eps:

                pp= solve_b(A,b/delta_t)

                p_new = p_new + pp
                p_new -= np.mean(p_new)
            print("t", t, "norma p", np.linalg.norm(b))

        u_prev = u_new
        v_prev = v_new
        p_prev = p_new
        t += delta_t
    plot_solution(u_prev, v_prev, p_prev, N,T)

def plot_solution(u, v, p, N,T, streamplot=True):
    # print(u.shape)
    # print(v.shape)
    # u = u[:, 1:]
    u = (u[:, :-1] + u[:, 1:]) / 2
    v = (v[1:, :] + v[:-1, :]) / 2
    # u=u[:,::-1]
    # v=v[:,::-1]
    # p=p.reshape((N,N))[:,::-1]

    # u = u.reshape(N * N, 1)[::-1].reshape(N, N)
    # v = v.reshape(N * N, 1)[::-1].reshape(N, N)
    # p = p[::-1]
    print(u.shape)
    u = u[::-1, ::]
    v = -v[::-1, ::]
    p = p.reshape((N, N))[::-1, ::]
    x = np.arange(h / 2, 1, h)
    y = np.arange(h / 2, 1, h)
    # y = np.arange(-1, -self.h/2, self.h)
    grid_x, grid_y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(10, 10))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.streamplot(grid_x, grid_y, u, v, color='black')
    plt.contourf(grid_x, grid_y, p.reshape((N, N)))
    plt.title(f"N = {N}, Eps = {eps}, Nu = {mu}, dt = {delta_t}", fontsize=20)

    #plt.savefig(f'data/N = {N}, Eps = {eps}, Nu = {mu},Time = {T} dt = {delta_t}.png')
    plt.show()



def div(u,v):
    global h
    global N
    b = np.zeros(N**2)
    for i in range(N):
        for j in range(N):
            b[i * N + j] = h * (u[i,j+1] - u[i,j] + v[i+1,j] - v[i,j])
    return b

def solve_speed(u_prev, v_prev, p_prev, N,t):
    global mu
    global h
    u_new = np.zeros((N, N+1)) #= u_prev
    v_new =np.zeros((N+1, N)) #= v_prev
    for i in range(N):
        for j in range(1,N):
                uc = u_prev[i,j]
                uw = u_prev[i, j-1]
                ue = u_prev[i,j+1]

                if i == 0:
                    un = 2 - uc
                else:
                    un = u_prev[i-1, j]
                if i == N-1:
                    us = -uc
                else:
                    us = u_prev[i+1, j]

                vnw = v_prev[i, j-1]
                vsw = v_prev[i+1, j-1]
                vne = v_prev[i, j]
                vse = v_prev[i+1,j]
                pe = p_prev[i * N + j]
                pw = p_prev[i * N + j - 1]
                gradU =  (-(((uw + uc) / 2) ** 2) + (((uc + ue)/2) ** 2) + ((vsw + vse)/2)*((uc + us)/2) - ((vnw + vne)/2 * (uc + un)/2))
                # gradU = 0.25 / h * (
                #         (uc + ue) * (uc + ue) - (uw + uc) * (uw + uc) - (vnw + vne) * (un + uc) + (
                #         vsw + vse) * (us + uc)un = 2 - uc
                # )

                u_new[i,j]= uc - t * (gradU / h + mu/(h**2) * (4 * uc - ue - un - us -uw) + (p_prev[i*N+j] - p_prev[i*N+j-1]) /h)
                # u_new[i, j] = uc - t * (
                #         gradU + mu / (h ** 2) * (4 * uc - uw - ue - us - un) + (pe - pw) * h)
    for i in range(1,N):
        for j in range(N):
            vc = v_prev[i,j]
            vn = v_prev[i - 1, j]
            vs = v_prev[i + 1, j]
            if j == 0:
                vw = -vc
            else:
                vw = v_prev[i, j-1]
            if j == N-1:
                ve = -vc
            else:
                ve = v_prev[i, j + 1]

            usw = u_prev[i, j]
            use = u_prev[i, j+1]
            une = u_prev[i-1, j+1]
            unw = u_prev[i-1, j]
            pn = p_prev[(i - 1) * N + j]
            ps = p_prev[i * N + j]
            gradV =  (((ve+vc)/2 * (une + use)/2) - (vc + vw)/2 * (unw + usw)/2 + ((vs + vc)/2)**2 - ((vn+vc)/2)**2)
            # gradV = 0.25 /h * (
            #         (une + use) * (ve + vc) - (unw + usw) * (vc + vw) - ((vn + vc)) ** 2 + (
            #     (vs + vc)) ** 2
            # )
            v_new[i, j] = vc - t * (gradV/h + mu/(h**2) * (4 * vc - ve - vn - vs - vw) + (p_prev[i * N + j] - p_prev[(i-1)*N +j]) / h)
            #v_new[i, j] = vc - delta_t * (gradV + mu / (h ** 2) * (4 * vc - vw - ve - vs - vn) + (ps - pn) * h )

    return u_new,v_new



def solve_b(A, b):

     res = np.linalg.solve(A,b)
     return res




def ij(i,j):
    global N
    return (i * N) + j

def create_A(N):
    A = np.zeros((N**2, N**2))
    for i in range(N):
        for j in range(N):
            counter = 0
            if i != 0:
            #     print('None')
            # else:
                A[ij(i, j), ij(i, j)-N] = -1
                counter += 1

            if j != 0:
            #     print('None')
            # else:
                A[ij(i,j), ij(i,j) - 1] = -1
                counter += 1

            if j != N-1:
            #     print('None')
            # else:
                A[ij(i, j), ij(i, j) + 1] = -1
                counter += 1

            if i != N-1:
            #     print('None')
            # else:
                A[ij(i, j), ij(i,j) + N ] = -1
                counter += 1
            A[ij(i,j), ij(i,j)] = counter
    return A



def main():
    global N
    p_prev = np.zeros((N**2))
    u_prev = np.zeros((N, N+1))
    for i in range(N+1):
        u_prev[0][i] = 1
    v_prev = np.zeros((N+1, N))
    b = np.ones(N**2)
    solver(10,p_prev,u_prev,v_prev,b)


mu = 0.01
N = 16
eps = 0.01
delta_t = 0.001
A = create_A(N)
h = 1 / N


if __name__ == '__main__':
    main()
