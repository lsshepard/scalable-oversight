import numpy as np, sys
sys.argv=['x']; exec(open('floor.py').read().split('w_dirty=')[0])
def br_int(P,beta,lam,G2,vals=np.arange(-3,4)):
    n=P.shape[0]; mu=P@pi_u; post=(P*pi_u)/mu[:,None]; K=post@P.T
    best=np.zeros(n)
    EG2=K@G2; EG22=K@(G2**2)
    for z in range(n):
        cost=[]
        for g in vals:
            c=beta*(g*g-2*g*EG2[z]+EG22[z])+lam*sum(post[z,u]*min(u+g,0)**2 for u in KU)
            cost.append(c)
        best[z]=vals[int(np.argmin(cost))]
    return best
for lamM in (0.5,1.5,3):
  for lam in (0.4,1,3):
    P=build(lamM,0.0); k=np.array(sorted(range(P.shape[0])))  # p=0: state index = total count order
    G=-(np.arange(P.shape[0])>=1).astype(float)   # hide-one
    for _ in range(50):
        Gn=br_int(P,1,lam,G)
        if np.allclose(Gn,G): break
        G=Gn
    print(f"plants~Poisson({lamM}) lam={lam}: hide-one best-response fixed point -> {G[:6]}")
