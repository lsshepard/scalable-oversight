import numpy as np
PI=np.array([.8,.15,.05]); MEAN_T_DIRTY=(0.15*1+0.05*2)/0.2
def run(plant_mean,init,b,seed,steps=8000,B=512,lr=0.02,sig=0.1,beta=1.,lam=1.,eps=0.02,NZ=14):
    rng=np.random.default_rng(seed)
    z=np.arange(NZ,dtype=float)
    th=np.stack([{'honest':z,'zero':0*z,'hide_one':np.maximum(z-1,0),'random':rng.uniform(0,3,NZ)}[init].copy() for _ in range(2)])
    if init=='random': th[1]=rng.uniform(0,3,NZ)
    for t in range(steps):
        u=rng.choice(3,size=B,p=PI); m=rng.poisson(plant_mean,size=(2,B)).clip(0,NZ-3)
        zi=u[None,:]+m; noise=rng.normal(size=(2,B))*sig
        F=th[np.arange(2)[:,None],zi]+noise; V=F-m
        r=np.empty((2,B))
        for i in range(2):
            r[i]=-beta*(V[i]-V[1-i])**2-lam*np.maximum(m[i]-F[i],0)**2+b*(u>0)*(u-V[i])-eps*F[i]
        for i in range(2):          # REINFORCE with per-state baseline
            g=np.zeros(NZ); cnt=np.zeros(NZ); base=np.zeros(NZ)
            np.add.at(cnt,zi[i],1); np.add.at(base,zi[i],r[i]); base=base/np.maximum(cnt,1)
            np.add.at(g,zi[i],(r[i]-base[zi[i]])*noise[i]/sig**2)
            th[i]+=lr*np.clip(g/np.maximum(cnt,1),-5,5)
    # evaluate mean policy: concealment on dirty originals
    u=rng.choice(3,size=200000,p=PI); m=rng.poisson(plant_mean,size=200000).clip(0,NZ-3)
    V=th[0][u+m]-m; d=u>0
    return (u[d]-V[d]).mean()/MEAN_T_DIRTY, (V[~d]).mean()
def ci(x): x=np.array(x); return x.mean(),1.96*x.std(ddof=1)/np.sqrt(len(x))
print("share of genuine misconduct concealed (mean +- 95% CI over 8 seeds); same reward formula, plants on/off")
for label,init,b in [("A. start colluding (report zero), no stake","zero",0.0),
                     ("B. start colluding (hide one), no stake","hide_one",0.0),
                     ("C. start honest, stake b=0.2","honest",0.2),
                     ("D. start random, stake b=0.2","random",0.2)]:
    for pm,name in ((0.0,"no perturbation"),(1.5,"hidden perturbation")):
        res=[run(pm,init,b,s) for s in range(8)]
        c,e=ci([r[0] for r in res]); k,_=ci([r[1] for r in res])
        print(f"{label:45s} {name:20s} concealed={100*c:6.1f}% +-{100*e:4.1f}   mean report on clean originals={k:+.2f}")
