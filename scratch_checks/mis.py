import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
rng=np.random.default_rng(1)
# states (u,m): view z=ku+m ; residual xi(u,m) unknown to designer. n players general.
def run(n,lamM,beta,lam,b,scale,M=10):
    pi=np.array([.8,.15,.05]); pm=poisson.pmf(np.arange(M+1),lamM); pm/=pm.sum()
    nz=M+3; xi=scale*rng.normal(size=(3,M+1))
    w=pi[:,None]*pm[None,:]                       # joint (u,m)
    zidx=np.arange(3)[:,None]+np.arange(M+1)[None,:]
    P=np.zeros((nz,3)); 
    for u in range(3): P[u:u+M+1,u]=pm
    mu=P@pi; post=(P*pi)/mu[:,None]; K=post@P.T
    D=np.sqrt(mu); ev=np.sort(np.linalg.eigvalsh((D[:,None]*K)/D[None,:]))[::-1]; gam=1-ev[1]
    S=b*np.array([0,1,1.]); a=post@S; sbar=mu@a; sig2=mu@(a-sbar)**2
    lbar=min(lam,beta*gam/2); Q=sig2/(beta*gam)+sbar**2/(lbar*0.8); rho2=np.sum(w*xi**2)
    def pot(x):
        G=x.reshape(n,nz); val=0; g=np.zeros_like(G)
        e=[G[i][zidx]+xi for i in range(n)]       # (u,m) arrays: xi+G
        Ee=[(ei*pm).sum(1) for ei in e]           # E[.|u]
        Ee2=[(ei**2*pm).sum(1) for ei in e]
        for i in range(n):
            for j in range(n):
                if i<j:
                    val+=beta/(n-1)*np.sum(pi*(Ee2[i]+Ee2[j]-2*Ee[i]*Ee[j]))
            # gradient wrt G_i(z): sum over (u,m) with z=u+m
            gi=np.zeros(nz)
            for j in range(n):
                if j!=i: np.add.at(gi,zidx,2*beta/(n-1)*w*(e[i]-Ee[j][:,None]))
            xneg=np.minimum(np.arange(3)[:,None]+e[i],0)
            val+=lam*np.sum(w*xneg**2); np.add.at(gi,zidx,2*lam*w*xneg)
            val+=mu@(a*G[i]); gi+=mu*a; g[i]=gi
        return val,g.ravel()
    r=minimize(pot,np.zeros(n*nz),jac=True,method='L-BFGS-B',options=dict(maxiter=50000,ftol=1e-16,gtol=1e-11))
    G=r.x.reshape(n,nz); h=G@mu; H=G-h[:,None]
    Ebar=np.mean([beta*gam*mu@H[i]**2+lbar*0.8*min(h[i],0)**2 for i in range(n)])
    return Ebar,32*Q+16*(2*beta+lam)*rho2,Q,rho2,np.abs(G-G[0]).max()
print("n lamM scale  Ebar  bound(mis)  Q  rho2  asym")
ok=True
for n in (2,3):
  for lamM in (0.7,2):
    for scale in (0,0.1,0.5):
      for b in (0,0.2):
        E,bd,Q,r2,asym=run(n,lamM,1,0.4,b,scale); ok&=E<=bd+1e-9
        if scale==0: ok&=E<=Q+1e-9
        print(n,lamM,scale,b,"%.4f %.4f %.4f %.4f %.1e"%(E,bd,Q,r2,asym))
print("all bounds hold:",ok)
