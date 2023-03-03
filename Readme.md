import numpy as np

def Bayesian_DOA_root(Y, search_area, etc):
    M, T = Y.shape
    K_hat = len(search_area)
    
    reslu = search_area[1] - search_area[0]
    search_mid_left = search_area - reslu/2
    search_mid_right = search_area + reslu/2
    
    a_search = search_area*np.pi/180.
    A_theta = np.exp(-1j*np.pi*np.outer(np.arange(0, M), np.sin(a_search)))
    
    # initialization
    a = 0.0001
    b = 0.0001
    d = 0.01
    maxiter = 500
    tol = 1e-4
    beta = 1
    delta = np.ones(K_hat)
    
    converged = False
    iter = 0
    
    while not converged:
        iter += 1
        delta_last = delta
        
        # calculate mu and Sigma
        Phi = A_theta
        V_temp = 1/beta*np.eye(M) + Phi*np.diag(delta)@Phi.T
        Vinv = np.linalg.inv(V_temp)
        Sigma = np.diag(delta) - np.diag(delta)@Phi.T@Vinv@Phi@np.diag(delta)
        mu = beta*Sigma@Phi.T@Y
        gamma1 = 1 - np.real(np.diag(Sigma))/delta
        
        # update delta
        temp = np.sum(mu*np.conj(mu), axis=1) + T*np.real(np.diag(Sigma))
        delta = (-T + np.sqrt(T**2 + 4*d*np.real(temp)))/(2*d)
        
        # update beta
        resid = Y - Phi@mu
        beta = (T*M + (a-1))/(b + np.linalg.norm(resid, 'fro')**2 + T/beta*np.sum(gamma1))
        
        # stopping criteria
        erro = np.linalg.norm(delta - delta_last)/np.linalg.norm(delta_last)
        if erro < tol or iter >= maxiter:
            converged = True
        
        # root-refining
        f = np.sqrt(np.sum(mu*np.conj(mu), axis=1))
        sort_ind = np.argsort(f)
        index_amp = sort_ind[-1:-(etc+1):-1]
        
        for j in range(len(index_amp)):
            ii = index_amp[j]
            mut = mu[ii,:]
            Sigmat = Sigma[:,ii]
            phi = mut@mut.conj().T + T*Sigmat[ii]
            tempind = np.arange(K_hat)
            tempind = np.delete(tempind, ii)
            Yti = Y - Phi[:,tempind]@mu[tempind,:]
            varphi = T*Phi[:,tempind]@(Sigmat[tempind]) - Yti@(mut.conj().T)
            z1 = np.arange(1, M)
            c = np.zeros(M)
            c[0] = M*(M-1)/2*phi
            c[1:] = z1*varphi[1:]
            
            # root method
            ro = np.roots(c)
            abs_root = np.abs(ro)
            indmin = np.argmin(np.abs(abs_root-1))
            angle_cand = np.arcsin(-np.angle(ro[indmin])/np.pi)/np.pi*180
            
            if angle_cand <= search_mid_right[ii] and angle_cand >= search_mid_left[ii]:
                search_area[ii] = angle_cand
                A_theta[:,ii] = np.exp(-1j*np.pi
