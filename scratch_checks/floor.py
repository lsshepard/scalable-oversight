import numpy as np
from scipy.stats import binom, poisson
from scipy.optimize import minimize
pi_u=np.array([0.8,0.15,0.05]); KU=np.arange(3)
def build(lamM,p,M=9):
    pm=poisson.pmf(np.arange(M+1),lamM); pm/=pm.sum()
    zs={}
    for ku in KU:
        for m in range(M+1):
            for s in range(m+1): zs.setdefault((ku+m,s),len(zs))
    P=np.zeros((len(zs),3))
    for ku in KU:
        for m in range(M+1):
            for s in range(m+1): P[zs[(ku+m,s)],ku]+=pm[m]*binom.pmf(s,m,p)
    keep=P@pi_u>1e-13; P=P[keep]
    return P
def solve(P,beta,lam,b,w,eps=0.0,G0=None):
    n=P.shape[0]; mu=P@pi_u; post=(P*pi_u)/mu[:,None]; K=post@P.T
    def pot(x):
        G1,G2=x[:n],x[n:]
        comp=mu@G1**2+mu@G2**2-2*mu@(G1*(K@G2))
        val=beta*comp; g1=2*beta*mu*(G1-K@G2); g2=2*beta*mu*(G2-K@G1)
        for G,g in ((G1,g1),(G2,g2)):
            x_=np.minimum(KU[None,:]+G[:,None],0)        # (z,u)
            val+=lam*np.sum(P*pi_u*x_**2); g+=2*lam*np.sum(P*pi_u*x_,axis=1)
            val+=b*np.sum(P*pi_u*w*G[:,None]); g+=b*(P*pi_u*w).sum(1)   # stake: gain from -G
            val+=eps*mu@G; g+=eps*mu
        return val,np.concatenate([g1,g2])
    x0=np.zeros(2*n) if G0 is None else G0
    r=minimize(pot,x0,jac=True,method='L-BFGS-B',options=dict(maxiter=20000,ftol=1e-15,gtol=1e-10))
    G=r.x[:n]
    U=np.sum(P*pi_u*w*(-G[:,None]))                    # stake-weighted under-reporting
    dirty=np.sum(P[:,1:]*pi_u[1:]*(-G[:,None]))/pi_u[1:].sum()
    clean=np.sum(P[:,0]*pi_u[0]*(-G))/pi_u[0]
    D=np.diag(np.sqrt(mu)); ev=np.sort(np.linalg.eigvalsh(D@K@np.linalg.inv(D)))[::-1]
    gam=1-ev[1]; Ad=post@w; md=pi_u@w; sig2=mu@(Ad-md)**2
    return dict(U=U,dirty=dirty,clean=clean,gam=gam,sig2=sig2,md=md,asym=np.abs(r.x[:n]-r.x[n:]).max(),G=r.x)
w_dirty=np.array([0,1,1.]); w_all=np.ones(3)
print("== three regimes, stake b=0.2 on dirty logs (w=1[dirty]), lamM=1.5, p=0.2")
P=build(1.5,0.2)
for name,(be,la) in dict(comparison_only=(1,1e-9),floor_only=(1e-9,0.4),combined=(1,0.4)).items():
    for wn,w in (("dirty",w_dirty),("all",w_all)):
        r=solve(P,be,la,0.2,w)
        print(f"{name:16s} stake={wn:5s} under-report|dirty={r['dirty']:9.3f} |clean={r['clean']:9.3f}")
print("== bound check: U <= b(sig2/(beta*gam)+md^2/(lam*p0)), lam<=beta*gam/2")
for lamM in (0.7,1.5,3):
  for p in (0,0.3,0.6,0.9):
    P=build(lamM,p)
    for w in (w_dirty,w_all):
        r=solve(P,1,0.4*0+min(0.4,0.5*solve(P,1,.1,0,w)['gam']),0.2,w); lam=min(0.4,0.5*r['gam'])
        bd=0.2*(r['sig2']/r['gam']+r['md']**2/(lam*0.8))
        print(f"lamM={lamM} p={p} md={r['md']:.2f} gam={r['gam']:.3f} sig2={r['sig2']:.4f} U={r['U']:.4f} bound={bd:.4f} ok={r['U']<=bd+1e-9} asym={r['asym']:.1e}")
print("== recognizability sweep (combined, b=0.2 dirty stake, lamM=1.5): fraction of genuine content concealed")
for p in (0,0.2,0.4,0.6,0.8,0.95,1.0):
    P=build(1.5,p); r=solve(P,1,0.4,0.2,w_dirty)
    meanT=(pi_u[1]*1+pi_u[2]*2)/pi_u[1:].sum()
    print(f"p={p:4.2f} sig2={r['sig2']:.3f} gam={r['gam']:.3f} under-report|dirty={r['dirty']:.3f} ({100*r['dirty']/meanT:.0f}% of genuine) over-shade|clean={r['clean']:.3f}")
print("== no stake: zero-loss set and frugality tie-break")
P=build(1.5,0.2); n=P.shape[0]
for c in (-0.5,0.0,0.5):
    x=np.full(2*n,c); 
    r=solve(P,1,0.4,0,w_all,G0=x)  # start at shift c
    print(f"start shift {c:+.1f}: converges to mean shift {(P@pi_u)@r['G'][:n]:+.4f}")
for eps in (1e-3,1e-2):
    r=solve(P,1,0.4,0,w_all,eps=eps,G0=np.full(2*n,0.5)); print(f"eps={eps}: mean shift {(P@pi_u)@r['G'][:n]:+.5f}, max|G|={np.abs(r['G']).max():.5f}")
