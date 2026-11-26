import numpy as np, itertools
from scipy.stats import binom, poisson
rng=np.random.default_rng(0)
def build(pi_u, pm, p):
    # u: genuine count k_u ~ pi_u ; m planted ~ pm ; s|m ~ Bin(m,p) recognised planted
    # z=(k,s). returns A[z,u]=P(z|u), mu_z, states
    zs={}; rows=[]
    for ku,pu in enumerate(pi_u):
        for m,q in enumerate(pm):
            for s in range(m+1):
                z=(ku+m,s); zs.setdefault(z,len(zs))
    P=np.zeros((len(zs),len(pi_u)))
    for ku,pu in enumerate(pi_u):
        for m,q in enumerate(pm):
            for s in range(m+1):
                P[zs[(ku+m,s)],ku]+=q*binom.pmf(s,m,p)
    pu=np.array(pi_u); mu=P@pu
    keep=mu>1e-14; P=P[keep]; mu=mu[keep]
    post=(P*pu)/mu[:,None]          # P(u|z)
    K=post@P.T                       # K[z,z']=P(z'|z)
    return P,pu,mu,post,K,zs
def analyse(pi_u,pm,p,b=1,beta=1,lam_alpha=0.0):
    P,pu,mu,post,K,zs=build(pi_u,pm,p)
    D=np.diag(np.sqrt(mu)); S=D@K@np.linalg.inv(D)
    ev=np.sort(np.linalg.eigvalsh((S+S.T)/2))[::-1]
    gamma=1-ev[1]
    ku=np.arange(len(pu)); d=-(ku-pu@ku); d=d/np.sqrt(pu@d**2)
    Ad=post@d; sig2=mu@Ad**2
    n=len(mu); M=beta*(np.eye(n)-K)+lam_alpha*np.eye(n)
    G=(b/2)*(np.linalg.pinv(M,rcond=1e-10)@Ad); G-= (mu@G if lam_alpha==0 else 0)
    C=mu@(Ad*G)
    # best-response iteration check
    G1=np.zeros(n);G2=np.zeros(n)
    for _ in range(20000):
        G1n=(b*Ad/2+beta*K@G2)/(beta+lam_alpha+1e-12); G2n=(b*Ad/2+beta*K@G1)/(beta+lam_alpha+1e-12)
        if lam_alpha==0: G1n-=mu@G1n; G2n-=mu@G2n
        G1,G2=G1n,G2n
    Cbr=mu@(Ad*G1)
    return gamma,sig2,C,Cbr, b*sig2/(2*(beta*gamma+lam_alpha))
pi_u=[0.8,0.15,0.05]
print("p_detect gamma sigma_d^2 C_formula C_bestresp bound")
for lamM in [1,3]:
  pm=poisson.pmf(np.arange(12),lamM); pm/=pm.sum()
  for p in [0,0.25,0.5,0.75,0.9,0.99]:
    print(lamM,p,*["%.4f"%x for x in analyse(pi_u,pm,p)])
# audit-only vs combined
pm=poisson.pmf(np.arange(12),3); pm/=pm.sum()
print("audit-only (beta=0, lam*alpha=0.05):",["%.3f"%x for x in analyse(pi_u,pm,0.25,beta=1e-9,lam_alpha=0.05)])
print("combined   (beta=1, lam*alpha=0.05):",["%.3f"%x for x in analyse(pi_u,pm,0.25,beta=1,lam_alpha=0.05)])
# cross Poincare check
P,pu,mu,post,K,zs=build(pi_u,pm,0.25)
D=np.diag(np.sqrt(mu)); ev=np.sort(np.linalg.eigvalsh(D@K@np.linalg.inv(D)))[::-1]; g=1-ev[1]
worst=9
for _ in range(20000):
    G1=rng.normal(size=len(mu)); G2=G1+rng.normal(size=len(mu))*rng.choice([0,.1,1])
    L=mu@G1**2+mu@G2**2-2*mu@(G1*(K@G2))
    m1,m2=mu@G1,mu@G2; rhs=g*(mu@(G1-m1)**2+mu@(G2-m2)**2)+(m1-m2)**2
    worst=min(worst,L/rhs)
print("min loss/bound ratio (should be >=1):",worst, "gamma",g)
