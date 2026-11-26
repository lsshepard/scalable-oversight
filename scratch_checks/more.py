import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
# generic finite model: states u with prob pi, T(u), context c(u); views z with P[z,u]
def ops(P,pi):
    mu=P@pi; post=(P*pi)/mu[:,None]; K=post@P.T; return mu,post,K
def gap(P,pi):
    mu,post,K=ops(P,pi); D=np.sqrt(mu); S=(D[:,None]*K)/D[None,:]
    ev=np.sort(np.linalg.eigvalsh((S+S.T)/2))[::-1]
    k=int(np.sum(ev>1-1e-9)); return 1-ev[k],k
def solve(P,pi,T,beta,lam,S,n=2,xi=None):
    nz=P.shape[0]; mu,post,K=ops(P,pi); a=post@S
    def pot(G):  # symmetric potential restricted is not enough; use 2 players
        G1,G2=G[:nz],G[nz:]
        val=beta*(mu@G1**2+mu@G2**2-2*mu@(G1*(K@G2))); g1=2*beta*mu*(G1-K@G2); g2=2*beta*mu*(G2-K@G1)
        for Gi,gi in ((G1,g1),(G2,g2)):
            x=np.minimum(T[None,:]+Gi[:,None],0); val+=lam*np.sum(P*pi*x**2); gi+=2*lam*np.sum(P*pi*x,axis=1)
            val+=mu@(a*Gi); gi+=mu*a
        return val,np.concatenate([g1,g2])
    r=minimize(pot,np.zeros(2*nz),jac=True,method='L-BFGS-B',options=dict(maxiter=50000,ftol=1e-16,gtol=1e-11))
    return r.x[:nz],mu,post
# (1) overlap bound gamma>=eta, count model
print("(1) gamma vs overlap eta")
for lamM in (0.5,1.5,3,6):
    M=25; pm=poisson.pmf(np.arange(M+1),lamM); pm/=pm.sum(); pi=np.array([.8,.15,.05])
    P=np.zeros((M+3,3))
    for ku in range(3): P[ku:ku+M+1,ku]=pm
    eta=P.min(axis=1).sum(); g,k=gap(P,pi); print(f" plants Poisson({lamM}): eta={eta:.3f} gamma={g:.3f} common-dim={k}")
# (6) two contexts; context B is 'signalled': always dirty (T>=1). stake blanket eps
print("(6) signalling context: under-report by context")
pm=poisson.pmf(np.arange(13),1.5); pm/=pm.sum(); M=12
# states: (ctx A,T=0),(A,1),(B,1),(B,2)
pi=np.array([.72,.08,.15,.05]); T=np.array([0,1,1,2.]); ctx=[0,0,1,1]
nz=2*(M+3); P=np.zeros((nz,4))
for s in range(4):
    for m in range(M+1): P[ctx[s]*(M+3)+int(T[s])+m,s]+=pm[m]
keep=P@pi>0; P=P[keep]
g,k=gap(P,pi); print(" gamma above context=%.3f, common-dim=%d"%(g,k))
for b in (0.02,0.2):
    G,mu,post=solve(P,pi,T,1,0.4,np.full(4,b))
    cz=np.array([0]*(M+3)+[1]*(M+3))[keep]
    for c in (0,1):
        w=mu*(cz==c); print(f"  blanket stake {b}: mean shift in context {'AB'[c]} = {w@G/w.sum():+.3f}")
# (2) coarse reports: binary original, integer reports F>=0
print("(2) integer reports, hide-one equilibrium condition lam < beta(1/cmax-2q0)")
def hideone_is_NE(p0,lamM,beta,lam,M=30):
    pm=poisson.pmf(np.arange(M+1),lamM); pm/=pm.sum(); pi=np.array([p0,1-p0]); T=np.array([0,1.])
    P=np.zeros((M+2,2)); P[:M+1,0]=pm; P[1:,1]=pm
    mu,post,K=ops(P,pi); G=-(np.arange(M+2)>=1).astype(float); ok=True
    for z in range(M+2):
        if mu[z]<1e-12: continue
        best=None
        for g in range(-z,4):
            c=beta*(g*g-2*g*(K@G)[z]+(K@G**2)[z])+lam*sum(post[z,u]*min(T[u]+g,0)**2 for u in (0,1))
            if best is None or c<best[0]-1e-12: best=(c,g)
        ok&=(best[1]==G[z])
    cmax=max(post[z,0] for z in range(1,M+2) if mu[z]>1e-12); return ok, beta*(1/cmax-2*pm[0])
for lam in (0.2,0.5,0.8,1.0,1.2):
    ok,thr=hideone_is_NE(0.8,1.5,1,lam); print(f" lam={lam}: hide-one NE={ok}  threshold={thr:.3f}")
