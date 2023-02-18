import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit, vmap, jacrev
import jax.scipy.special as jsp
from jax.scipy.stats import uniform, multivariate_normal, norm, beta
from jax.numpy import heaviside
from scipy.optimize import minimize
from pmagpy import pmag,ipmag
from pmagpy import contribution_builder as cb
from itertools import combinations
from adjustText import adjust_text
from matplotlib.collections import LineCollection
import warnings
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import truncnorm
from functools import lru_cache
from jax.flatten_util import ravel_pytree
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

@jit
def SGG_cdf(x,mu,sigma,p,q):
    """
    Calculates the cumulative density function of the skew
    generalized gaussian (SGG) distribution (Egli, 2003)
    
    Inputs
    ------
    x: float or array
    x value at which to evaluate SGG distribution
    
    mu: float
    mu parameter (controls location) of SGG distribution
    
    sigma: float
    sigma parameter (controls scale) of SGG distribution
    
    p: float
    p parameter (controls kurtosis) of SGG distribution
    
    q: float
    q parameter (controls skewness) of SGG distribution
    
    Returns
    -------
    G: float or array
    Value of SGG cdf at x
    """
    z=(x-mu)/sigma
    u=jnp.log((jnp.exp(q*z)+jnp.exp(z/q))/2)
    #Regularized Upper Incomplete Gamma Function (1/p,0.5*abs(u)**p)
    inc_gam_part=jsp.gammaincc(1/p,(jnp.abs(u)**p)/2)
    heaviside_part=heaviside(u,0.5)
    sign_part=jnp.sign(u)/2
    G=1-heaviside(q,0.5)+jnp.sign(q)*(heaviside_part-sign_part*inc_gam_part)
    return(G)

@jit
def SGG_pdf(x, mu, sigma, p, q):
    """
    Calculates the cumulative density function of the skew
    generalized gaussian (SGG) distribution (Egli, 2003)
    
    Inputs
    ------
    x: float or array
    x value at which to evaluate SGG distribution
    
    mu: float
    mu parameter (controls location) of SGG distribution
    
    sigma: float
    sigma parameter (controls scale) of SGG distribution
    
    p: float
    p parameter (controls kurtosis) of SGG distribution
    
    q: float
    q parameter (controls skewness) of SGG distribution
    
    Returns
    -------
    g: float or array
    Value of SGG pdf at x
    """
    z=(x-mu)/sigma
    prefac=1/(2**(1+1/p)*sigma*jnp.exp(jsp.gammaln(1+1/p)))
    skewpart=jnp.abs(q*jnp.exp(q*z)+jnp.exp(z/q)/q)/(jnp.exp(q*z)+jnp.exp(z/q))
    exponential=jnp.exp(-0.5*jnp.abs(jnp.log((jnp.exp(q*z)+jnp.exp(z/q))/2))**p)
    return(prefac*skewpart*exponential)

@jit
def construct_pred_data(Bs,Ts,cs,mus,sds,ps,qs):
    """
    Uses the TROUT model parameters and the 
    temperatures/fields of the demag data to
    predict data from the TROUT model.
    
    Inputs
    ------
    Bs: kx3 array
    Set of cartesian field directions for the 
    different components in the TROUT model. 
    
    Ts: length n array
    Set of temperatures/fields used to demag
    the specimen.
    
    cs: length k array
    Magnitude of each component
    
    mus: length k array
    SGG "mu" parameter for each component
    
    sds: length k array
    SGG "sigma" parameter for each component
    
    ps: length k array
    SGG "p" parameter for each component
    
    qs: length k array
    SGG "q" parameter for each component
    
    Returns
    -------
    data_pred: nx3 array
    Predicted data from the TROUT model
    """
    n=Ts.shape[0] #number of demag data
    k=cs.shape[0] #number of components
    Bs_scaled=jnp.empty(Bs.shape) #Scaled Bs
    Fs=jnp.empty((k,n)) #Scaled Fs
    data_pred=jnp.zeros((n,3)) #Predicted data
    
    for i in range(k):
        #Create SGG cdf values.
        Fs=Fs.at[i].set(1-SGG_cdf(Ts,mus[i],sds[i],ps[i],qs[i]))
        for j in range(3):
            #Scaled Bs is cs*Bs
            Bs_scaled=Bs_scaled.at[i,j].set(cs[i]*Bs[i,j])
            #predicted data is cs*Bs*Fs
            data_pred=data_pred.at[:,j].add(Bs_scaled[i,j]*Fs[i])
    return(data_pred)

@jit
def cdf_misfit(pars,data,Ts):
    """
    Calculates misfit from a set of data to an SGG cdf
    for initialization of TROUT model
    
    Inputs
    ------
    pars: length 4 array
    mu, sd, p_star,q_star parameters for SGG distribution
    
    data: length n array
    data to fit to SGG cdf
    
    Ts: length n array
    Temperatures at which cdf is evaluated 
    
    Returns
    -------
    like: float
    likelihood of cdf given data
    """
    mu,sd,p_star,q_star=pars
    #Calculate p and q parameters
    p=jnp.exp(p_star)
    q=2/jnp.pi*jnp.arctan(1/q_star)
    
    #Calculate predicted data
    pred_data=1-SGG_cdf(Ts,mu,sd,p,q)
    
    #Calculate likelihood (misfit to data)
    like=len(pred_data)*jnp.log(jnp.sum((data-pred_data)**2))/2-uniform.logpdf(p_star,0,5)-uniform.logpdf(q_star,-5,10)
    return(like)

def import_direction_data(wd):
    """
    Imports directions from magic tables
    
    Inputs
    ------
    wd: string
    working directory containing magic tables
    
    Returns
    -------
    data: pandas DataFrame
    specimen table with added samples/sites
    
    sample_data: pandas DataFrame
    sample table
    
    site_data: pandas DataFrame
    site table
    """
    
    status,data=cb.add_sites_to_meas_table(wd)
    sample_data=pd.read_csv(wd+'samples.txt',skiprows=1,sep='\t')
    site_data=pd.read_csv(wd+'sites.txt',skiprows=1,sep='\t')
    return(data,sample_data,site_data)

@jit
def scale_xs(xs_base,expon):
    """
    Scales temperatures/AF fields/microwave power
    used in demagnetization experiment to make
    TROUT model easier to fit.
    
    Inputs
    ------
    xs_base: length n array
    temperatures/AF fields to scale
    
    expon: float
    exponent to scale data by. If expon > 0 then
    data are scaled by their distance from the 
    highest temperature/field.
    
    Returns
    -------
    xs_scaled: length n array
    scaled x values. 
    """
    exponexpon=jnp.exp(-jnp.abs(expon))
    xs_scaled=(jnp.heaviside(expon,0)*jnp.max(xs_base)-(jnp.heaviside(expon,0)*2-1)*xs_base)**exponexpon
    xs_scaled=jnp.heaviside(expon,0)*jnp.max(xs_scaled)-(jnp.heaviside(expon,0)*2-1)*xs_scaled
    return(xs_scaled)

@jit
def unscale_xs(xs_scaled,expon):
    """
    Inverse function of scale_xs. Corrects scaled
    temperature/fields back to original values.
    
    Inputs
    ------
    xs_scaled: length n array
    scaled x values.
    
    expon: float
    exponent to scale data by. If expon > 0 then
    data are scaled by their distance from the 
    highest temperature/field.
    
    Returns
    -------
    xs_base: length n array
    original x values
    """
    exponexpon=jnp.exp(-jnp.abs(expon))
    xs_base=(jnp.heaviside(expon,0)*jnp.max(xs_scaled)-(jnp.heaviside(expon,0)*2-1)*xs_scaled)**exponexpon
    xs_base=jnp.heaviside(expon,0)*jnp.max(xs_base)-(jnp.heaviside(expon,0)*2-1)*xs_base
    return(xs_base)


@jit
def bestfit_tanh(pars,x,y):
    """
    Calculates misfit to a tanh function
    
    Inputs:
    -------
    pars: length 2 array
    scale and location parameters of
    tanh function
    
    x: length n array 
    x values
    
    y: length n array
    y values
    
    Returns
    -------
    diff: float
    sum of squared differences from tanh 
    function.
    """
    xp=scale_xs(x,pars[1])
    tanhfunc=(1-jnp.tanh(pars[0]*(xp-jnp.min(xp))/(jnp.max(xp)-jnp.min(xp))-pars[0]/2))/2
    diff=jnp.sum((y-tanhfunc)**2)
    return diff

@jit
def calculate_gradient_covariances(zijd_data,Ts_data,sigma,psi):
    """
    
    Calculates covariances of gradients of vector demagnetization
    data, given a diagonal gaussian noise given by "sigma" and
    a scale dependent noise (attitude) given by "psi"

    """
    n=zijd_data.shape[0]
    #Array of original covariances
    Cs=jnp.empty((3,3,n))
    
    #Populate covariance matrices
    for l in range(len(zijd_data)):
        BB=jnp.outer(zijd_data[l],zijd_data[l])
        B2=jnp.linalg.norm(zijd_data[l])**2
        C=jnp.identity(3)*(sigma**2+B2*psi**2)-BB*psi**2
        Cs=Cs.at[:,:,l].set(C)
        
    #Get Cs for summing covariances
    Cs_i=Cs[:,:,1:-1]
    Cs_plus=Cs[:,:,2:]
    Cs_minus=Cs[:,:,:-2]
    
    #Get spacing for multiplying variances
    T_is=Ts_data[1:-1]
    Ts_plus=Ts_data[2:]
    Ts_minus=Ts_data[:-2]
    
    hs=T_is-Ts_minus
    hd=Ts_plus-T_is

    #Create covariances of numerical gradient
    Cs_grad=jnp.empty(Cs.shape)
    Cs_grad=Cs_grad.at[:,:,1:-1].set((Cs_plus*hs**4+Cs_minus*hd**4+Cs_i*jnp.abs(hd**2-hs**2)**2)/(hs*hd*(hd+hs))**2)
    Cs_grad=Cs_grad.at[:,:,0].set((Cs[:,:,1]+Cs[:,:,0])/(Ts_data[1]-Ts_data[0])**2)
    Cs_grad=Cs_grad.at[:,:,-1].set((Cs[:,:,-2]+Cs[:,:,-1])/(Ts_data[-1]-Ts_data[-2])**2)
    return(Cs_grad.T)

@jit
def construct_pred_diffs(Bs,Ts,cs,mus,sds,ps,qs):
    """
    Constructs predicted data values from the TROUT
    model.

    Inputs
    ------
    Bs: K by 3 array
    Cartesian direction vector for each of the
    K components.

    Ts: length N array
    Temperatures for thermal demag or coercivities 
    for AF demag

    cs: length K array
    Magnitudes of each component

    mus: length k array
    "mu" parameter of SGG distribution for each 
    component

    sds: length k array
    "sd" parameter of SGG distribution for each component

    ps: length k array
    "p" parameter of SGG distribution for each component

    qs: length k array
    "q" parameter of SGG distribution for each component

    Returns
    -------
    data_pred: nx3 array
    Predicted demagnetization data.
    """
    n=Ts.shape[0]
    k=cs.shape[0]
    Bs_scaled=jnp.empty(Bs.shape)
    Fs=jnp.empty((k,n))
    data_pred=jnp.zeros((n,3))
    
    for i in range(k):
        Fs=Fs.at[i].set(jnp.nan_to_num(SGG_pdf(Ts,mus[i],sds[i],ps[i],qs[i]),0))
        for j in range(3):
            Bs_scaled=Bs_scaled.at[i,j].set(cs[i]*Bs[i,j])
            data_pred=data_pred.at[:,j].add(Bs_scaled[i,j]*Fs[i])
    return(data_pred)


@jit
def like_func_grad(zijd_data,zijd_diffs,Ts_data,Bs,cs,mus,sds,p_stars,q_stars,sigma,psi):
    """
    Likelihood function using gradients of demagnetization data (deprecated)
    """
    ps=jnp.exp(p_stars)
    qs=2/jnp.pi*jnp.arctan(1/q_stars)
    Trange=jnp.ptp(Ts_data)
    Ts_data/=Trange
    mus/=Trange
    sds/=Trange
    pred_diffs=construct_pred_diffs(Bs,Ts_data,cs,mus,sds,ps,qs)
    Cs_grad=calculate_gradient_covariances(zijd_data,Ts_data,sigma,psi)
    llike_=0
    for l in range(len(zijd_data)):
        llike_+=multivariate_normal.logpdf(pred_diffs[l],zijd_diffs[l],Cs_grad[l])
    return(llike_)
    
@jit
def post_func_grad(par_dict,zijd_data,zijd_diffs,Ts_data):
    """
    Posterior distribution for using gradients of demagnetization data
    (deprecated).
    """
    Bs=(par_dict['Ms'].T/jnp.linalg.norm(par_dict['Ms'],axis=1)).T
    cs=jnp.linalg.norm(par_dict['Ms'],axis=1)
    c_norm=jnp.sum(cs)/vds
    lp=prior_func_sigma_psi(
        c_norm,par_dict['sigma'])
    ll=like_func_grad(
        zijd_data,zijd_diffs,Ts_data,Bs,cs,
        par_dict['mus'],par_dict['sds'],
        par_dict['p_stars'],par_dict['q_stars'],
        par_dict['sigma'],par_dict['psi'])
    return(jnp.nan_to_num(ll+lp,nan=-jnp.inf))


def prepare_specimen_data(specimen,data,sample_data,site_data,normalize=True):
    """
    Process data from a specimen for use with TROUT. Rotates into geographical
    coordinates if applicable.

    Inputs
    ------
    specimen: string
    name of specimen being processed

    data: pandas DataFrame
    magic specimens table imported to DataFrame

    sample_data: pandas DataFrame
    magic samples table imported to DataFrame

    site_data: pandas DataFrame
    magic sites table imported to DataFrame

    normalize: bool
    Whether or not to normalize the temperatures or coer80ivities for better 
    fitting with TROUT (should be left at True)

    Returns
    -------
    zijd_data: nx3 array
    Set of demagnetization data the TROUT model will be applied to

    Ts_base: length n array
    Original Temperature steps (or coercivities).

    Ts_data: length n array
    Scaled Temperature steps or coercivities

    Ts: length 100 array
    Interpolation on range of temperature steps

    datatype: string
    Type of demagnetization, currently "AF" or "Thermal"

    expon: float
    Exponent parameter used to scale the data.
    """
    specframe=data[(data.specimen==specimen)&((data.method_codes.str.contains('LP-DIR'))|(data.method_codes.str.contains('LT-T-Z')|(data.method_codes.str.contains('LT-NO'))|(data.method_codes.str.contains('LP-NO'))))]
    sample=specframe['sample'].iloc[0]
    site=specframe['site'].iloc[0]
    specdata_dir=np.array([specframe['dir_dec'].values,specframe['dir_inc'].values,specframe['magn_moment'].values]).T
    sampleframe=sample_data[sample_data['sample']==sample]
    try:
        sampleframe.dropna(subset=['azimuth'],inplace=True)
        specdata_dir_corr=np.array([pmag.dogeo(specdata_dir[i,0],specdata_dir[i,1],sampleframe['azimuth'].values[0],sampleframe['dip'].values[0]) for i in range(len(specdata_dir))])
        specdata_dir_corr=np.append(specdata_dir_corr,specdata_dir[:,2][:,np.newaxis],axis=1)
        specdata=pmag.dir2cart(specdata_dir_corr)
    except:
        specdata=pmag.dir2cart(specdata_dir)
    zijd_data=specdata/max(np.sqrt(np.sum(specdata**2,axis=1)))
    
    if specframe.method_codes.str.contains('LP-DIR-AF').iloc[0]:
        datatype='af'
        Ts_base=specframe.treat_ac_field.values*1e3

    
    elif (specframe.method_codes.str.contains('LP-DIR-T').iloc[0])|(specframe.method_codes.str.contains('LT-NO').iloc[0]):
        Ts_base=specframe.treat_temp.values-273
        datatype='thermal'
   
    else:
        datatype='unspecified'
        Ts_base=specframe.treat_temp.values-273
        max_data=max(Ts_base)
        Ts_base=Ts_base/max(Ts_base)
    
    flipped_data=np.flip(zijd_data,axis=0)
    diff_data=np.diff(flipped_data,axis=0)
    appended_data=np.append([[0,0,0]],diff_data,axis=0)
    norm_data=np.linalg.norm(appended_data,axis=1)+np.linalg.norm(zijd_data[-1])
    cumulative_data=np.cumsum(norm_data)
    Ms=np.flip(cumulative_data)
    Ms_scaled=Ms/max(Ms)
    if normalize==True:
        minimizer=minimize(bestfit_tanh,x0=[1.,0.1],args=(Ts_base,Ms_scaled),method='L-BFGS-B',bounds=([1,100],[-10,10]),options={'eps':np.sqrt(np.finfo('float32').eps)})
        expon=minimizer.x[1]
        Ts_data=np.array(scale_xs(Ts_base,expon))        

    else:
        Ts_data=Ts_base
        expon=0
    Ts=np.linspace(min(Ts_data),max(Ts_data),100)
    
    return(zijd_data,Ts_base,Ts_data,Ts,datatype,expon)

def simple_fit(zijd_data,Ts_data,break_points):
    """
    A simple way of getting a best guess for TROUT initialization
    (deprecated)
    """
    endpoints=[]
    dirs=[]
    cs=[]
    for i in range(len(break_points)-1):
        zijd_data_trunc=zijd_data[(Ts_data<=break_points[i+1])&(Ts_data>=break_points[i])]
        endpoints.append(zijd_data_trunc[0])
        if i==(len(break_points)-2):
            zijd_data_trunc=np.append(zijd_data_trunc,[[0,0,0]],axis=0)
            endpoints.append(zijd_data_trunc[-1])
        direction=zijd_data_trunc[0]-zijd_data_trunc[-1]
        c=np.linalg.norm(zijd_data_trunc[0]-zijd_data_trunc[-1])
        dirs.append(direction/c)
        cs.append(c)
    d2=0
    for i in range(len(dirs)):
        zijd_data_trunc=zijd_data[(Ts_data<=break_points[i+1])&(Ts_data>=break_points[i])]
        if i!=0:
            zijd_data_trunc=zijd_data_trunc[1:]
        line=endpoints[i+1]-endpoints[i]
        linept=endpoints[i]-zijd_data_trunc
        
        d2s=np.linalg.norm(np.cross(line,linept),axis=1)**2/np.linalg.norm(line)**2
        ts=-np.dot(linept,line)/np.linalg.norm(line)**2
        d2s[ts<0]=np.linalg.norm(zijd_data_trunc[ts<0]-endpoints[i],axis=1)**2
        d2s[ts>1]=np.linalg.norm(zijd_data_trunc[ts>1]-endpoints[i+1],axis=1)**2
        d2+=np.sum(d2s)
        
    return(np.array(dirs),np.array(cs),d2)

def pca_fit(zijd_data,Ts_data,break_points):
    """
    Performs a PCA analysis on a specimen with multiple components.Attempts to
     minimize the misfit by finding the closest point to the intersection of
    each component's segment. The "break_points" parameter specifies the points
    at which the component "changes" (assuming no overlap)

    Inputs
    ------
    zijd_data: nx3 array
    demagnetization data for a specimen

    Ts_data: length n array
    scaled temperature data for a specimen. 

    break_points: length (k-1) array
    Temperatures/coercivities at which component changes

    Returns
    -------
    dirs: length k array
    Directions of principal components

    cs: length k array
    Magnitudes of principal components

    d2: float
    Sum of squared distances of points to principal axis.
    """
    #Set up PCA fits
    cs=[]
    dirs=np.empty((len(break_points)-1,3))
    means=np.empty((len(break_points)-1,3))
    vs=[]
    for i in range(len(break_points)-1):
        zijd_data_trunc=zijd_data[(Ts_data<=break_points[i+1])&(Ts_data>=break_points[i])]
        if i==(len(break_points)-2):
            zijd_data_trunc=np.append(zijd_data_trunc,[[0,0,0]],axis=0)
        
        if len(zijd_data_trunc)==2:

            cs.append(np.linalg.norm(np.diff(zijd_data_trunc,axis=0)))
            dirs[i]=np.diff(np.flip(zijd_data_trunc,axis=0),axis=0)/cs[i]
            vs.append(np.array([cs[i]/2*dirs[i],-cs[i]/2*dirs[i]]))
            means[i]=(zijd_data_trunc[0]+zijd_data_trunc[1])/2
        else:
            pca=PCA(n_components=3)
            pca=pca.fit(zijd_data_trunc)
            length, vector=pca.explained_variance_[0], pca.components_[0]
            
            vals=pca.transform(zijd_data_trunc)[:,0]
            v = np.outer(vals,vector)
            vs.append(v)
            means[i]=pca.mean_
            cs.append(np.linalg.norm((pca.mean_+v[-1])-(pca.mean_+v[0])))
            dirs[i]=((pca.mean_+v[0])-(pca.mean_+v[-1]))/cs[i]
    
    #Correct PCA fits so that lines intersect.
    for i in range(len(dirs)-1):
        offdist=np.cross(dirs[i],dirs[i+1])
        offdist/=np.linalg.norm(offdist)
        closestpt=np.linalg.solve(np.array([dirs[i],-dirs[i+1],offdist]).T,(means[i+1]+vs[i+1][0])-(means[i]+vs[i][-1]))
        cs[i+1]+=closestpt[1]
        cs[i]-=closestpt[0]
        
    endpoints=np.empty((len(break_points),3))
    for i in range(len(dirs)):
        endpoints[i]=np.sum(np.transpose(dirs[i:].T*cs[i:]),axis=0)
    endpoints[-1]=[0,0,0]
    d2=0
    for i in range(len(dirs)):
        zijd_data_trunc=zijd_data[(Ts_data<=break_points[i+1])&(Ts_data>=break_points[i])]
        if i!=0:
            zijd_data_trunc=zijd_data_trunc[1:]
        line=endpoints[i+1]-endpoints[i]
        linept=endpoints[i]-zijd_data_trunc
        
        d2s=np.linalg.norm(np.cross(line,linept),axis=1)**2/np.linalg.norm(line)**2
        ts=-np.dot(linept,line)/np.linalg.norm(line)**2
        d2s[ts<0]=np.linalg.norm(zijd_data_trunc[ts<0]-endpoints[i],axis=1)**2
        d2s[ts>1]=np.linalg.norm(zijd_data_trunc[ts>1]-endpoints[i+1],axis=1)**2
        d2+=np.sum(d2s)
        
    return(np.array(dirs),np.array(cs),d2)

def find_best_dist(zijd_data,Ts_data,break_points):
    """
    Finds the best fitting SGG distribution to a set of demagnetization data, 
    assuming no overlap between components which change at temperatures or
    coercivities specified in "break_points".

    Inputs
    ------
    zijd_data: nx3 array
    demagnetization data for a specimen

    Ts_data: length n array
    scaled temperature data for a specimen. 

    break_points: length (k-1) array
    Temperatures/coercivities at which component changes

    Returns
    -------
    mus: length k array
    Best fitting SGG "mu" parameters

    sds: length k array
    Best fitting SGG "sd" parameters

    ps: length k array
    Best fitting SGG "p" parameters

    qs: length k array
    Best fitting SGG "q" parameters
    """
    #Minimize SGG distribution
    mus=[]
    sds=[]
    p_stars=[]
    q_stars=[]
    for i in range(len(break_points)-1):
        zijd_data_trunc=zijd_data[(Ts_data<=break_points[i+1])&(Ts_data>=break_points[i])]
        
        zijd_data_trunc=zijd_data_trunc-zijd_data_trunc[-1]
        flipped_zijd=np.flip(zijd_data_trunc)
        mags_trunc=np.append(0,np.cumsum(np.linalg.norm(np.diff(flipped_zijd,axis=0),axis=1)))
        mags_trunc/=np.amax(mags_trunc)
        mags_trunc=np.flip(mags_trunc)
        Ts_data_trunc=Ts_data[(Ts_data<=break_points[i+1])&(Ts_data>=break_points[i])]
        sorted_mags=mags_trunc[np.argsort(mags_trunc)]
        trunc_Ts_scaled=(Ts_data_trunc-break_points[i])/(break_points[i+1]-break_points[i])
        sorted_Ts=trunc_Ts_scaled[np.argsort(mags_trunc)]
        start_sd=(jnp.interp(0.16,sorted_mags,sorted_Ts)-jnp.interp(0.84,sorted_mags,sorted_Ts))/2
        start_mu=jnp.interp(0.5,sorted_mags,sorted_Ts)
        start_p_star=2.
        start_q_star=0.01

        start_pars=[start_mu,start_sd,start_p_star,start_q_star]
        result=minimize(cdf_misfit,start_pars,jac=jacrev(cdf_misfit),args=(mags_trunc,trunc_Ts_scaled),method='BFGS',options={'eps':np.sqrt(np.finfo('float32').eps)})
        
        mus.append(result.x[0]*(break_points[i+1]-break_points[i])+break_points[i])
        sds.append(result.x[1]*(break_points[i+1]-break_points[i]))
        p_stars.append(result.x[2])
        q_stars.append(result.x[3])
    return(np.array(mus),np.array(sds),np.array(p_stars),np.array(q_stars))

def find_naive_fit(zijd_data,Ts_data,n_components,bpoints=None,anchored=False):
    """
    Attempts to find a best fitting TROUT solution to a set of demagnetization
    data assuming no unblocking temperature overlap.

    Inputs
    ------
    zijd_data: nx3 array
    demagnetization data for a specimen

    Ts_data: length n array
    scaled temperature data for a specimen. 

    n_components: int
    number of components (k) expected for this specimen.

    bpoints: len (k-1) array
    Forces fit to assume a change in component at a particular temperature or 
    coercivity (if set to None, this is already chosen). This should probably 
    be set to None.

    anchored: bool
    When True, performs the TROUT fit to all data. TROUT assumes that at high
    temperatures, the magnetization goes to zero. With Anchored=False, the 
    origin is moved to the final demagnetization step so this requirement is
    not met. With Anchored=True an additional component may be needed for 
    specimens where the magnetization does not trend towards the origin.

    Returns
    -------
    Bs: nx3 array
    Best guess field directions

    cs: nx3 array
    Best guess component magnitudes

    mus: length k array
    Best guess SGG "mu" parameters

    sds: length k array
    Best guess SGG "sd" parameters

    ps: length k array
    Best guess SGG "p" parameters

    qs: length k array
    Best guess SGG "q" parameters
    """
    d2=np.inf
    numbers = Ts_data[1:-1]
    Bs=[]
    cs=[]

    if anchored==False:
        zijd_data-=zijd_data[-1]
    if bpoints==None:
        for item in combinations(numbers, n_components-1):
            break_points=[Ts_data[0]]+sorted(item)+[Ts_data[-1]]
            Bs_new,cs_new,d2_new=pca_fit(zijd_data,Ts_data,break_points)
            
            if d2_new<d2:
                d2=d2_new
                Bs=Bs_new
                cs=cs_new
                break_points_final=break_points
        
    else:
        break_points=[Ts_data[0]]+sorted(bpoints)+[Ts_data[-1]]
        Bs_new,cs_new,d2=pca_fit(zijd_data,Ts_data,break_points)
        Bs=Bs_new
        cs=cs_new
        break_points_final=break_points
    mus,sds,p_stars,q_stars=find_best_dist(zijd_data,Ts_data,break_points_final)
    
    return(Bs,cs,mus,sds,p_stars,q_stars)

@jit
def like_func_notol(Ms,Ts,Bs,cs,mus,sds,p_stars,q_stars):
    n=Ts.shape[0]
    ps=jnp.exp(p_stars)
    qs=2/jnp.pi*jnp.arctan(1/q_stars)
    Bs=(Bs.T/jnp.linalg.norm(Bs,axis=1)).T
    data_pred=construct_pred_data(Bs,Ts,cs,mus,sds,ps,qs)
    llike_=0
    for l in range(n):
        llike_+=jnp.sum((data_pred[l]-Ms[l])**2)
    llike_=-3*n*jnp.log(llike_)/2
    return(llike_)

@jit
def like_func_sigma_psi(Ms,Ts,Bs,cs,mus,sds,p_stars,q_stars,sigma,psi):
    """
    Likelihood function for the TROUT model, gives the log likelihood of a set 
    of demagnetization data, given the model parameters.

    Inputs
    ------
    Ms: nx3 array
    Demagnetization data.

    Ts: length n array
    Temperatures at which demag data is evaluated.

    Bs: length k array
    Field directions for each component of demag data

    cs: length k array
    Magnitudes of each component of demag data

    mus: length k array
    "mu" parameters of SGG distribution for each component

    sds: length k array
    "s" parameters of SGG distribution for each component

    p_stars: length k array
    "p_star" parameters (can be transformed to "p") of SGG distribution for
    each component

    q_stars: length k array
    "q_star" parameters (can be transformed to "q") of SGG distribution for
    each component

    sigma: float>0
    Standard deviation of cartesian noise on demagnetization data.

    psi: float>0
    Angular uncertainty (in radians) of demagnetization data.

    Returns
    -------
    llike_: float
    log-likelihood of parameters given data.

    """
    #Shapes of data/model
    n=Ms.shape[0]
    k=cs.shape[0]
    m=Ms.shape[1]

    #Transform SGG Parameters
    ps=jnp.exp(p_stars)
    qs=2/jnp.pi*jnp.arctan(1/q_stars)

    #Predict mean values of data from model
    data_pred=construct_pred_data(Bs,Ts,cs,mus,sds,ps,qs)
    
    llike_=0
    for l in range(n):
        #Calculate Covariance matrix for this data point
        BB=jnp.outer(Ms[l],Ms[l])
        B2=jnp.linalg.norm(Ms[l])**2
        C=jnp.identity(3)*(sigma**2+B2*psi**2)-BB*psi**2

        #Log likelihood is probability of observed data given covariance matrix
        #and predicted mean data values
        llike_+=multivariate_normal.logpdf(data_pred[l],Ms[l],C)

    #Return log-likelihood
    return(llike_)

@jit
def like_func_sigma_chi(Ms,Ts,Bs,cs,mus,sds,p_stars,q_stars,sigma,chi):
    """
    Deprecated likelihood function (assuming all angular uncertainty in the
    x-y plane).
    """
    n=Ms.shape[0]
    k=cs.shape[0]
    m=Ms.shape[1]
    ps=jnp.exp(p_stars)
    qs=2/jnp.pi*jnp.arctan(1/q_stars)
    data_pred=construct_pred_data(Bs,Ts,cs,mus,sds,ps,qs)  
    llike_=0
    for l in range(n):
        Mval=jnp.cross(jnp.array([0,0,1]),Ms[l])
        BB=jnp.outer(Mval,Mval)
        C=jnp.identity(3)*(sigma**2)+BB*chi**2
        Cinv=jnp.linalg.inv(C)
        diff=data_pred[l]-Ms[l]
        llike_+=-diff.T@Cinv@diff/2-jnp.log(jnp.linalg.det(2*jnp.pi*C))/2
    return(llike_)

@jit
def alt_prior_sigma_psi(Ts_data,cs,mus,sds,p_stars,q_stars,sigma):
    """
    Prior used for the TROUT model. The prior is calculated using the overlap
    between two SGG distributions.

    Inputs
    ------
    Ts_data: length n array
    Temperatures at which demag data is evaluated. (This is included in the 
    prior only to set a range for the acceptable bounds of mu)

    Bs: length k array
    Field directions for each component of demag data

    cs: length k array
    Magnitudes of each component of demag data

    mus: length k array
    "mu" parameters of SGG distribution for each component

    sds: length k array
    "s" parameters of SGG distribution for each component

    p_stars: length k array
    "p_star" parameters (can be transformed to "p") of SGG distribution for
    each component

    q_stars: length k array
    "q_star" parameters (can be transformed to "q") of SGG distribution for
    each component

    sigma: float>0
    Standard deviation of cartesian noise on demagnetization data.

    Returns
    -------
    lp: float
    log prior distribution
    """

    #Transform SGG parameters
    ps=jnp.exp(p_stars)
    qs=2/np.pi*jnp.arctan(1/q_stars)

    #Set of xs to evaluate
    xs=jnp.linspace(jnp.min(Ts_data),jnp.max(Ts_data),1000)

    #Probability density function
    pdfs=jnp.empty((len(cs),1000))
    
    #Start with log prior of 0 and add to it
    lp=0
    for k in range(len(cs)):
        #Evaluate pdf over temperature range
        pdf=cs[k]*SGG_pdf(xs,mus[k],sds[k],ps[k],qs[k])
        pdfs=pdfs.at[k].set(pdf)
    
    #Find all combinations of two pdfs and calculate their overlap
    for ks in combinations(range(len(pdfs)),2):
        ks=jnp.array(ks)
        pdf_pair=pdfs[ks]
        #Minimum of the two pdfs at all temperatures
        cmin=np.amin(jnp.array([cs]).T[ks])
        overlap=jnp.amin(pdf_pair,axis=0)
        
        #Calculate the overlap coefficient
        overlap_sum=jnp.trapz(overlap,xs)/cmin
        #Prior is a beta distribution
        lp+=beta.logpdf(overlap_sum,1,10)

    #Prior on sigma is 1/sigma
    lp-=jnp.log(sigma)
    
    #Set bounds for SGG parameters
    Tmin=jnp.min(Ts_data)
    Tmax=jnp.max(Ts_data)
    lp+=sum(uniform.logpdf(mus,Tmin,Tmax-Tmin))
    lp+=sum(uniform.logpdf(sds,0,(Tmax-Tmin)*10/6))
    lp+=sum(uniform.logpdf(p_stars,0,5))
    lp+=sum(uniform.logpdf(q_stars,-5,10))
    lp+=sum(uniform.logpdf(cs,0,2))

    #Constraint that mu_0<mu_1<...<mu_k - avoids label degeneracy
    lp+=sum(uniform.logpdf(jnp.diff(mus),0,Tmax-Tmin))

    return(lp)
        
        

@jit
def prior_func_sigma_psi(c_norm,sigma):
    lp=norm.logpdf(jnp.amax(jnp.array([c_norm,1])),1,0.1)-jnp.log(sigma)
    return(lp)

@jit
def prior_func_notol(c_norm):
    lp=norm.logpdf(c_norm,1,0.1)
    return(lp)


@jit
def post_func_sigma_chi(pars,zijd_data,Ts_data):
    par_dict=par_vec_to_named_pars_sigma_chi(pars)
    Tmax=jnp.max(Ts_data)
    Tmin=jnp.min(Ts_data)
    norms=jnp.sum(jnp.linalg.norm(par_dict['Bs'],axis=1))
    lp=prior_func_sigma_chi(
        Tmax,Tmin,norms,par_dict['mus'],par_dict['sds'],par_dict['p_stars'],
        par_dict['q_stars'],par_dict['cs'],par_dict['sigma'],par_dict['chi'])
    ll=like_func_sigma_chi(
        zijd_data,Ts_data,par_dict['Bs'],par_dict['cs'],par_dict['mus'],
        par_dict['sds'],par_dict['p_stars'],par_dict['q_stars'],par_dict['sigma'],
        par_dict['chi'])
    return(jnp.nan_to_num(ll+lp,nan=-np.inf))

@jit
def post_func_sigma_psi(par_dict,zijd_data,Ts_data):
    """
    Posterior distribution function used in the TROUT model. Finds the relative
    probability of a set of TROUT parameters, given the demag data.

    Inputs
    ------
    par_dict: dictionary
    dictionary of parameters supplied to the posterior function

    zijd_data: nx3 array
    Demagnetization data

    Ts_data:
    Temperatures/coercivities at which demag data evaluated

    Returns
    -------
    lpost:
    log posterior distribution (plus a constant)

    """
    #We sample from a 3D vector called Ms which is Bs*cs. It has magnitude c 
    #and direction B.
    Bs=(par_dict['Ms'].T/jnp.linalg.norm(par_dict['Ms'],axis=1)).T
    cs=jnp.linalg.norm(par_dict['Ms'],axis=1)


    #Calculate Prior
    lp=alt_prior_sigma_psi(
        Ts_data,cs,par_dict['mus'],par_dict['sds'],
        par_dict['p_stars'],par_dict['q_stars'],
        par_dict['sigma'])
    
    ll=like_func_sigma_psi(
        zijd_data,Ts_data,Bs,cs,
        par_dict['mus'],par_dict['sds'],
        par_dict['p_stars'],par_dict['q_stars'],
        par_dict['sigma'],par_dict['psi'])
    lpost=jnp.nan_to_num(ll+lp,nan=-np.inf)
    return(lpost)

@jit
def post_func_notol(par_dict,zijd_data,Ts_data,vds):
    """
    Posterior function assuming all data have uniform gaussian noise
    (Deprecated)
    """
    Bs=(par_dict['Ms'].T/jnp.linalg.norm(par_dict['Ms'],axis=1)).T
    cs=jnp.linalg.norm(par_dict['Ms'],axis=1)
    c_norm=jnp.sum(cs)/vds
    lp=prior_func_notol(c_norm)
    ll=like_func_notol(zijd_data,Ts_data,Bs,cs,
                       par_dict['mus'],par_dict['sds'],
                       par_dict['p_stars'],par_dict['q_stars'])
    return(jnp.nan_to_num(ll+lp,nan=-np.inf))

@jit
def like_func_angle_notol(Ms,Ts,Bs,cs,mus,sds,p_stars,q_stars):
    """
    Deprecated
    """
    n=Ts.shape[0]
    ps=jnp.exp(p_stars)
    qs=2/jnp.pi*jnp.arctan(1/q_stars)
    data_pred=construct_pred_data(Bs,Ts,cs,mus,sds,ps,qs)
    
    data_pred_mag=jnp.linalg.norm(data_pred,axis=1)
    data_mag=jnp.linalg.norm(Ms,axis=1)
    dot_products=0
    magnitude_distances=0
    for l in range(n):
        dot_products+=jnp.nan_to_num(jnp.dot(data_pred[l],Ms[l])/(data_pred_mag[l]*data_mag[l]),0)
        magnitude_distances+=(data_mag[l]-data_pred_mag[l])**2
    llike_= -n*(jnp.log(n-dot_products)+jnp.log(magnitude_distances)/2)
        
    return(llike_)

@jit
def post_func_angle_notol(par_dict,zijd_data,Ts_data,vds):
    """
    Deprecated
    """
    Bs=(par_dict['Ms'].T/jnp.linalg.norm(par_dict['Ms'],axis=1)).T
    cs=jnp.linalg.norm(par_dict['Ms'],axis=1)
    #c_norm=jnp.sum(cs)/vds
    #lp=prior_func_notol(c_norm)
    ll=like_func_angle_notol(zijd_data,Ts_data,Bs,cs,
                       par_dict['mus'],par_dict['sds'],
                       par_dict['p_stars'],par_dict['q_stars'])
    return(jnp.nan_to_num(ll,nan=-np.inf))


@jit
def prior_func(Tmax,Tmin,norms,mus,sds,p_stars,q_stars,cs):
    """
    Deprecated
    """
    lp=sum(uniform.logpdf(mus,Tmin,Tmax-Tmin))+sum(uniform.logpdf(sds,0,(Tmax-Tmin)*10/6))+sum(uniform.logpdf(p_stars,-5,10))+sum(uniform.logpdf(q_stars,-5,10))+sum(uniform.logpdf(cs,0,2))-norms
    return(lp)

@jit
def post_func(pars,zijd_data,Ts_data):
    par_dict=par_vec_to_named_pars(pars)

    Tmax=jnp.max(Ts_data)
    Tmin=jnp.min(Ts_data)
    norms=jnp.sum(jnp.linalg.norm(par_dict['Bs'],axis=1))
    lp=prior_func(Tmax,Tmin,norms,par_dict['mus'],par_dict['sds'],par_dict['p_stars'],par_dict['q_stars'],par_dict['cs'])
    ll=like_func_notol(zijd_data,Ts_data,par_dict['Bs'],par_dict['cs'],par_dict['mus'],par_dict['sds'],par_dict['p_stars'],par_dict['q_stars'])
    return(jnp.nan_to_num(ll+lp,nan=-np.inf))

#Function cacheing is done through @lru_cache
@lru_cache(maxsize=None)
def create_unravel_funcs(n_components,post_func):
    """
    Data conversion for the a particular posterior distribution function. Many 
    packages for minimization and MCMC require parameters to be input as a 
    single array, losing the structure of e.g. Bs being a set of k length 3 
    unit vectors. This function remembers the structural relationship between 
    an array and a dictionary and converts between the two. The relationship is
    cached in memory.

    Inputs
    ------
    n_components: int
    Number of components

    post_func: function
    Posterior function

    Returns
    -------
    unravel_func: function
    Function to turn array into dictionary of parameters

    nlp: function
    Function to calculate log posterior from an array
    """

    #Create a parameter dictionary
    par_dict={
        'Ms':np.empty((n_components,3),dtype='float32'),
        'mus':np.empty(n_components,dtype='float32'),
        'sds':np.empty(n_components,dtype='float32'),
        'p_stars':np.empty(n_components,dtype='float32'),
        'q_stars':np.empty(n_components,dtype='float32')}
    
    #We need sigma,psi parameters if our posterior function has them
    if post_func in [post_func_sigma_psi,post_func_grad]:
        par_dict['sigma']=1e-1
        par_dict['psi']=np.radians(2)

    #"Ravels" the dictionary into an array and generates an inverse function
    pars,unravel_func=ravel_pytree(par_dict)
    post_func_a=post_func

    #If using gradients generate a log-posterior function including gradients
    if post_func==post_func_grad:
        @jit
        def nlp(pars,zijd_data,zijd_diffs,Ts_data,scales):
            par_dict=unravel_func(pars*scales)
            return -post_func_a(par_dict,zijd_data,zijd_diffs,Ts_data)

    #Otherwise generate a log posterior not including gradients
    else:
        @jit
        def nlp(pars,zijd_data,Ts_data,scales):
            par_dict=unravel_func(pars*scales)
            return -post_func_a(par_dict,zijd_data,Ts_data)
    
    return(unravel_func,nlp)


#Deprecated conversion functions
@jit
def par_vec_to_named_pars_sigma_chi(pars):
    n_components=int((len(pars)-2)/7)
    Ms=jnp.empty((n_components,3))
    mus=jnp.empty(n_components)
    sds=jnp.empty(n_components)
    deltas=jnp.empty(n_components)
    epsilons=jnp.empty(n_components)
    for n in range(n_components):
        Ms=Ms.at[n].set([pars[3*n],pars[3*n+1],pars[3*n+2]])
        mus=mus.at[n].set(pars[3*n_components+n])
        sds=sds.at[n].set(pars[4*n_components+n])
        deltas=deltas.at[n].set(pars[5*n_components+n])
        epsilons=epsilons.at[n].set(pars[6*n_components+n])
    sigma=pars[-2]
    chi=pars[-1]
    return({'Ms':Ms,'mus':mus,'sds':sds,'p_stars':deltas,'q_stars':epsilons,'sigma':sigma,'chi':chi})

def named_pars_to_par_vec_sigma_chi(pars,n_components):
    parvec=[]
    parnames=['mus','sds','p_stars','q_stars']
    for i in range(n_components):
        for j in range(3):
            parvec.append(pars['Ms'][i,j])

    for l in range(len(parnames)):
        for i in range(n_components):
            parvec.append(pars[parnames[l]][i])
    parvec.append(pars['sigma'])
    parvec.append(pars['chi'])

    return(np.array(parvec))



def calc_overlap_range(par_dict,Ts_base,expon,desired_fraction=0.95):
    """
    Calculates the crossover temperatures and lower and upper bounds of the 
    mixed region (See TROUT paper).

    Inputs
    ------
    par_dict: dict
    dictionary of best-fitting parameters output by the TROUT model.

    Ts_base: length n array
    Original temperatures/coercivities of demag experiment
    
    expon: float
    Exponent for scaling of temperatures

    desired_fraction: float
    Desired proportion of unblocking carried by one component. The default 
    value of 0.95 indicates that 95% of the unblocking is uncontaminated by 
    another component.

    Returns
    -------
    pdfs: 10000 x k array
    SGG Probability density functions

    crossovers: k x k array
    Matrix of crossover points between data. crossovers[i,j] represents the 
    crossover between component i and component j. There can be at most two 
    crossover points, crossover[j,i] will be different to crossover[i,j] in 
    this case.

    lowers: k x k array
    Matrix of lower bounds of mixed regions between data. lowers[i,j] 
    and lowers[j,i] give the lower bound of the mixed region where the
    proportions of both components i and j are greater than 1-desired_fraction.
    (i.e. the two components are overlapping) lowers[i,i] gives the lower bound
    for the region where the proportion of component i is greater than
    desired_fraction (i.e. where that component is unblocking solely).

    uppers: k x k array
    Specified in the same way as the "lowers" parameter, but for upper bounds
    on all respective regions.
    """

    #Interpolate distributions on basis of Ts_base
    xs_base=np.linspace(min(Ts_base),max(Ts_base),10000)
    xs_scaled=scale_xs(xs_base,expon)
    pdfs=[]
    exponexpon=jnp.exp(-jnp.abs(expon))

    #Calculate pdfs
    for i in range(len(par_dict['mus'])):
        p=np.exp(par_dict['p_stars'][i])
        q=2/np.pi*np.arctan(1/par_dict['q_stars'][i])
        pdf=par_dict['cs'][i]* \
            SGG_pdf(xs_scaled,par_dict['mus'][i],par_dict['sds'][i],p,q)
        
        #Reparameterize in terms of base Ts
        pdf*=exponexpon*(jnp.heaviside(expon,0)*jnp.max(xs_base) \
            -(jnp.heaviside(expon,0)*2-1)*xs_base)**(exponexpon-1)
        pdf=jnp.nan_to_num(pdf)
        pdfs.append(pdf)

    
        
   
    pdfs=np.array(pdfs)
    pdfs=pdfs[:,:-1] #Sometimes can get weird edge behaviour near to last value
    xs_scaled=xs_scaled[:-1]
    xs_base=xs_base[:-1]
    
    overlaps=pdfs/np.sum(pdfs,axis=0)

    
    overlaps=np.nan_to_num(overlaps,0)

    crossovers=np.full((len(par_dict['mus']),len(par_dict['mus'])),np.nan)
    nonzero=np.any(pdfs>1/np.ptp(xs_scaled)/1e3,axis=0)
    lowers=np.full((len(par_dict['mus']),len(par_dict['mus'])),np.nan)
    uppers=np.full((len(par_dict['mus']),len(par_dict['mus'])),np.nan)

    #Loop through all combinations of distributions
    for ij in combinations(range(len(par_dict['mus'])), 2):
        #Get indices
        i=ij[0]
        j=ij[1]

        #Calculate where the crossover point is
        overlap_pair=(overlaps[i],overlaps[j])
        increasing_equal=((overlap_pair[0][:-1]>overlap_pair[1][:-1])&(overlap_pair[0][1:]<overlap_pair[1][1:]))
        decreasing_equal=((overlap_pair[0][:-1]<overlap_pair[1][:-1])&(overlap_pair[0][1:]>overlap_pair[1][1:]))
        #Ignore points where both distributions are zero 
        # (proportion of 0/0, undefined.)
        nonzero=np.all(np.array(overlap_pair)>1/np.ptp(xs_scaled)/1e3,axis=0)
        #Filter for non zero regions where points cross over
        temp_filter=(increasing_equal|decreasing_equal)&nonzero[1:]&nonzero[:-1]
        crossover=(xs_base[1:]+xs_base[:-1])/2
        #Find mixed region
        multi_filter=np.all(np.array(overlap_pair)>(1-desired_fraction),axis=0)&nonzero
        multi_filter=multi_filter.at[-1].set(False)
        #Differencing booleans gives you the start and end points of the region
        multi_ranges=np.diff(multi_filter,prepend=False)
        
        multi_temps=xs_base[multi_ranges]
        multi_temps=np.reshape(multi_temps,(int(len(multi_temps)/2),2))
        
        #If there is any overlap, get CT and MR
        if (len(crossover[temp_filter])>0)&(len(multi_temps)>0):
            crossovers[i,j]=crossover[temp_filter][0]
            crossovers[j,i]=crossover[temp_filter][-1]
            lowers[i,j]=multi_temps[0][0]
            lowers[j,i]=multi_temps[-1][0]
            uppers[i,j]=multi_temps[0][1]
            uppers[j,i]=multi_temps[-1][1]

        #Otherwise, try to obtain the first and last temps where the proportion
        #is >0.5, the MR becomes the region between those and the crossovers
        #are set to the center of that region. 
        else:
            try:
                lasttemp=xs_base[(overlap_pair[0]>0.5)&(pdfs[i]>1/np.ptp(xs_scaled)/1e3)][-1]
                firsttemp=xs_base[(overlap_pair[1]>0.5)&(pdfs[j]>1/np.ptp(xs_scaled)/1e3)][0]

                crossovers[i,j]=(firsttemp+lasttemp)/2
                crossovers[j,i]=(firsttemp+lasttemp)/2
                lowers[i,j]=lasttemp
                lowers[j,i]=lasttemp
                uppers[i,j]=firsttemp
                uppers[j,i]=firsttemp
            except:
                pass
            
            
        
        for i in range(len(overlaps)):
            fraction=overlaps[i]
            pdf=pdfs[i]
            #Calculate where there is a component being demagnetized
            single_component=(fraction>desired_fraction)&(pdf>1/np.ptp(xs_scaled)/1e3)
            single_component=np.append(False,single_component)
            single_component[-1]=False
            #Calculate ranges where this component is active
            component_ranges=np.diff(single_component)
            component_temps=xs_base[component_ranges]
            component_temps=np.reshape(component_temps,(int(len(component_temps)/2),2))
            #Set lowers[i,i] to lower end of component range.
            if len(component_temps)>0:
                component_temp_best=component_temps[np.argmax(np.diff(component_temps,axis=1))]
                lowers[i][i]=component_temp_best[0]
                uppers[i][i]=component_temp_best[1]
    

    return(pdfs,crossovers,lowers,uppers)


@jit
def par_vec_to_named_pars(pars):
    n_components=int(len(pars)/8)
    Bs=jnp.empty((n_components,3))
    cs=jnp.empty(n_components)
    mus=jnp.empty(n_components)
    sds=jnp.empty(n_components)
    deltas=jnp.empty(n_components)
    epsilons=jnp.empty(n_components)
    for n in range(n_components):
        Bs=Bs.at[n].set([pars[3*n],pars[3*n+1],pars[3*n+2]])
        cs=cs.at[n].set(pars[3*n_components+n])
        mus=mus.at[n].set(pars[4*n_components+n])
        sds=sds.at[n].set(pars[5*n_components+n])
        deltas=deltas.at[n].set(pars[6*n_components+n])
        epsilons=epsilons.at[n].set(pars[7*n_components+n])
    return({'Bs':Bs,'cs':cs,'mus':mus,'sds':sds,'p_stars':deltas,'q_stars':epsilons})

def named_pars_to_par_vec(pars,n_components):
    parvec=[]
    parnames=['cs','mus','sds','p_stars','q_stars']
    for i in range(n_components):
        for j in range(3):
            parvec.append(pars['Bs'][i,j])

    for l in range(len(parnames)):
        for i in range(n_components):
            parvec.append(pars[parnames[l]][i])
    return(np.array(parvec))

def likefunc(params,Ms,Ts):
    return -like_func_notol(Ms,Ts,params['Bs'],params['cs'],params['mus'],params['sds'],params['p_stars'],params['q_stars'])

def jacobian(Ms,Ts,params):
    return jacrev(likefunc)(params,Ms,Ts)

@jit
def nlp_sigma_chi(pars,Ms,Ts,scales):
    pars=pars*scales
    par_dict=par_vec_to_named_pars_sigma_chi(pars)
    lp=jnp.log(par_dict['sigma'])
    ll=-like_func_sigma_chi(Ms,Ts,par_dict['Bs'],par_dict['cs'],par_dict['mus'],par_dict['sds'],par_dict['p_stars'],par_dict['q_stars'],par_dict['sigma'],par_dict['chi'])
    return(ll+lp)
@jit
def nlp_sigma_chi_jac(pars,Ms,Ts,scales):
    return(jacrev(nlp_sigma_chi)(pars,Ms,Ts,scales))

@jit
def nlp_notol(pars,Ms,Ts,scales):
    pars=pars*scales
    par_dict=par_vec_to_named_pars(pars)
    lp=jnp.log(par_dict['sigma'])
    ll=-like_func_notol(Ms,Ts,par_dict['Bs'],par_dict['cs'],par_dict['mus'],par_dict['sds'],par_dict['p_stars'],par_dict['q_stars'])
    return(ll+lp)
@jit
def nlp_jac(pars,Ms,Ts,scales):
    return(jacrev(nlp)(pars,Ms,Ts,scales))

def emcee_fit(zijd_data,Ts_data,par_dict,n_components,n_samples,anchored=False):
    if anchored==False:
        zijd_data-=zijd_data[-1]
    pars=named_pars_to_par_vec(par_dict,n_components)
    Trange=Ts_data[-1]-Ts_data[0]
    weights=[]
    for i in range(n_components):
        weights+=[5e-2,5e-2,5e-2]
    for i in range(n_components):
        weights+=[5e-2]
    for i in range(n_components):
        weights+=[Trange/40]
    for i in range(n_components):
        weights+=[Trange/24]
    for i in range(n_components):
        weights+=[0.25]
    for i in range(n_components):
        weights+=[0.25]

    ndim=len(pars)
    nwalkers=2*ndim
    
    pos=np.array(pars)+weights*np.random.randn(nwalkers,ndim)
    
    lp_final=-np.inf
    n_tries=0
    while (lp_final==-np.inf)&(n_tries<10):
        lp=0
        for p in range(len(pos)):
            par_dict2=par_vec_to_named_pars(pos[p])
            par_dict2['mus']=np.abs(par_dict2['mus'])
            par_dict2['mus'][par_dict2['mus']>=Ts_data[-1]]=Ts_data[-1]-0.01
            par_dict2['mus'][par_dict2['mus']<Ts_data[0]]=Ts_data[0]+0.01
            par_dict2['sds']=np.abs(par_dict2['sds'])
            par_dict2['sds'][par_dict2['sds']>=(Ts_data[-1]-Ts_data[0])*10/6]=(Ts_data[-1]-Ts_data[0])*10/6-0.01
            par_dict2['cs']=np.abs(par_dict2['cs'])
            par_dict2['cs'][par_dict2['cs']>2]=1.99
            pos[p]=named_pars_to_par_vec(par_dict2,n_components)
            lp+=prior_func(Trange,0.,1.,par_dict2['mus'],par_dict2['sds'],par_dict2['p_stars'],par_dict2['q_stars'],par_dict2['cs'])
        lp_final=lp
        
        if n_tries==9:
            print('Warning: could not find good start conditions, initial guess may be outside prior bounds')
            warnings.warn('Warning: could not find good start conditions, initial guess may be outside prior bounds')
        n_tries+=1
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, post_func, args=(zijd_data,Ts_data)
    )

    sampler.run_mcmc(pos,n_samples,progress=True,)
    
    samples=sampler.get_chain()
    dir_list=np.empty((len(samples)*nwalkers,n_components,3))
    c_list=np.empty((len(samples)*nwalkers,n_components))
    mu_list=np.empty((len(samples)*nwalkers,n_components))
    sd_list=np.empty((len(samples)*nwalkers,n_components))
    delta_list=np.empty((len(samples)*nwalkers,n_components))
    epsilon_list=np.empty((len(samples)*nwalkers,n_components))


    for n in range(n_components):
        Bs_curr=np.array([samples[:,:,3*n].flatten(),samples[:,:,3*n+1].flatten(),samples[:,:,3*n+2].flatten()]).T
        Bs_curr=(Bs_curr.T/np.linalg.norm(Bs_curr,axis=1)).T
        dir_list[:,n,:]=Bs_curr
        c_list[:,n]=samples[:,:,3*n_components+n].flatten()
        mu_list[:,n]=samples[:,:,4*n_components+n].flatten()
        sd_list[:,n]=samples[:,:,5*n_components+n].flatten()
        delta_list[:,n]=samples[:,:,6*n_components+n].flatten()
        epsilon_list[:,n]=samples[:,:,7*n_components+n].flatten()
        
    return(sampler,dir_list,c_list,mu_list,sd_list,delta_list,epsilon_list)

def emcee_fit_sigma_chi(zijd_data,Ts_data,par_dict,n_components,n_samples,anchored=False):
    if anchored==False:
        zijd_data-=zijd_data[-1]
    pars=named_pars_to_par_vec_sigma_chi(par_dict,2)
    Trange=Ts_data[-1]-Ts_data[0]
    weights=[]
    for i in range(n_components):
        weights+=[5e-2,5e-2,5e-2]
    for i in range(n_components):
        weights+=[5e-2]
    for i in range(n_components):
        weights+=[Trange/40]
    for i in range(n_components):
        weights+=[Trange/24]
    for i in range(n_components):
        weights+=[0.25]
    for i in range(n_components):
        weights+=[0.25]
    weights+=[0.01]
    weights+=[np.radians(0.5)]
    
    ndim=len(pars)
    nwalkers=2*ndim
    
    pos=np.array(pars)+weights*np.random.randn(nwalkers,ndim)
    
    lp_final=-np.inf
    n_tries=0
    while (lp_final==-np.inf)&(n_tries<10):
        lp=0
        for p in range(len(pos)):
            par_dict2=par_vec_to_named_pars_sigma_chi(pos[p])
            par_dict2['mus']=np.abs(par_dict2['mus'])
            par_dict2['mus'][par_dict2['mus']>=Ts_data[-1]]=Ts_data[-1]-0.01
            par_dict2['mus'][par_dict2['mus']<Ts_data[0]]=Ts_data[0]+0.01
            par_dict2['sds']=np.abs(par_dict2['sds'])
            par_dict2['sds'][par_dict2['sds']>=(Ts_data[-1]-Ts_data[0])*10/6]=(Ts_data[-1]-Ts_data[0])*10/6-0.01
            par_dict2['cs']=np.abs(par_dict2['cs'])
            par_dict2['cs'][par_dict2['cs']>2]=1.99
            par_dict2['sigma']=np.abs(par_dict2['sigma'])
            if par_dict2['sigma']>10:
                par_dict2['sigma']=9.99
            par_dict2['chi']=np.abs(par_dict2['chi'])
            if par_dict2['chi']>np.pi/18:
                par_dict2['chi']=np.pi/18-0.001
            pos[p]=named_pars_to_par_vec_sigma_chi(par_dict2,n_components)
            lp+=prior_func_sigma_chi(Trange,0.,1.,par_dict2['mus'],par_dict2['sds'],par_dict2['p_stars'],par_dict2['q_stars'],par_dict2['cs'],par_dict2['sigma'],par_dict2['chi'])
        lp_final=lp
        
        if n_tries==9:
            print('Warning: could not find good start conditions, initial guess may be outside prior bounds')
            warnings.warn('Warning: could not find good start conditions, initial guess may be outside prior bounds')
        n_tries+=1
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, post_func_sigma_chi, args=(zijd_data,Ts_data)
    )

    sampler.run_mcmc(pos,n_samples,progress=True,)
    
    samples=sampler.get_chain()
    dir_list=np.empty((len(samples)*nwalkers,n_components,3))
    c_list=np.empty((len(samples)*nwalkers,n_components))
    mu_list=np.empty((len(samples)*nwalkers,n_components))
    sd_list=np.empty((len(samples)*nwalkers,n_components))
    delta_list=np.empty((len(samples)*nwalkers,n_components))
    epsilon_list=np.empty((len(samples)*nwalkers,n_components))
    sigma_list=np.empty((len(samples)*nwalkers))
    chi_list=np.empty((len(samples)*nwalkers))

    for n in range(n_components):
        Bs_curr=np.array([samples[:,:,3*n].flatten(),samples[:,:,3*n+1].flatten(),samples[:,:,3*n+2].flatten()]).T
        Bs_curr=(Bs_curr.T/np.linalg.norm(Bs_curr,axis=1)).T
        dir_list[:,n,:]=Bs_curr
        c_list[:,n]=samples[:,:,3*n_components+n].flatten()
        mu_list[:,n]=samples[:,:,4*n_components+n].flatten()
        sd_list[:,n]=samples[:,:,5*n_components+n].flatten()
        delta_list[:,n]=samples[:,:,6*n_components+n].flatten()
        epsilon_list[:,n]=samples[:,:,7*n_components+n].flatten()
    sigma_list=samples[:,:,-2].flatten()
    chi_list=samples[:,:,-1].flatten()
        
    return(sampler,dir_list,c_list,mu_list,sd_list,delta_list,epsilon_list,sigma_list,chi_list)


def plot_zijd_data(ax,zijd_data,Ts_base,shared_axis=0):
    """
    Creates Zijderveld plot of a set of demagnetization data. 

    Inputs
    ------
    zijd_data: n x 3 array
    Set of demagnetization data to be plotted.

    Ts_base: length n array
    Temperatures/coercivities each demagnetization step was treated with.

    shared_axis: int
    Index of axis (x, y or z) is shared between both zijderveld projections.

    Returns
    -------
    None
    """
    ax.axvline(0,color='k',lw=1,zorder=-1)
    ax.axhline(0,color='k',lw=1,zorder=-1)
    axes=np.arange(3)
    unshared=axes[axes!=shared_axis]
    xylines=ax.plot(zijd_data[:,shared_axis],zijd_data[:,unshared[0]],'k',zorder=0)
    xzlines=ax.plot(zijd_data[:,shared_axis],zijd_data[:,unshared[1]],'k',zorder=0)
    xypoints=ax.scatter(zijd_data[:,shared_axis],zijd_data[:,unshared[0]],c='k',zorder=1)
    xzpoints=ax.scatter(zijd_data[:,shared_axis],zijd_data[:,unshared[1]],c='w',edgecolor='k',marker='s',zorder=1)
    zrange=np.max(zijd_data[:,unshared[1]])-np.min(zijd_data[:,unshared[1]])
    yrange=np.max(zijd_data[:,unshared[0]])-np.min(zijd_data[:,unshared[0]])
    if yrange>zrange:
        textindex=unshared[0]
    else:
        textindex=unshared[1]
    ax.axis('equal')
    ax.set_ylim(np.flip(ax.get_ylim()))

    grads=np.gradient(zijd_data,axis=0)
    gradgrads=np.gradient(grads,axis=0)
    gradgrads[gradgrads==-np.inf]=-1e38
    gradgrads[gradgrads==np.inf]=1e38
    gradgrads=np.transpose(gradgrads.T/np.linalg.norm(gradgrads[:,[shared_axis,textindex]],axis=1))


    n_clusters=min(int(len(zijd_data)/2),8)
    clusters=KMeans(n_clusters,random_state=0).fit_predict(zijd_data[:,[shared_axis,textindex]])
    indices=[]
    for i in range(n_clusters):
        points=np.where(clusters==i)[0]
        indices.append(points[int(len(points)/2)])

    for j in indices:
        diff=-gradgrads[j]
        tic_loc=zijd_data[j]+diff*0.02
        if diff[shared_axis]>0:
            ha='left'
        else:
            ha='right'
        if diff[textindex]>0:
            va='top'
        else:
            va='bottom'
        ax.annotate(str(int(Ts_base[j])),(tic_loc[shared_axis],tic_loc[textindex]),fontsize=14,ha=ha,va=va)
        ax.plot([zijd_data[j,shared_axis],tic_loc[shared_axis]],[zijd_data[j,textindex],tic_loc[textindex]],'k',lw=1,zorder=-1)
    else:
        pass
    ax.relim()

@jit
def get_B_ratios(Bs,Ts_data,cs,mus,sds,p_stars,q_stars,expon):
    """
    Calculates ratios of components demagnetizing at a set of temperatures. 
    Interpolates demagnetization data using the TROUT model.

    Inputs
    ------
    Bs: k x 3 array
    Field directions

    Ts_data: length n array
    Scaled temperatures/coercivities of actual demag data

    cs: length K array
    Magnitudes of each component

    mus: length k array
    "mu" parameter of SGG distribution for each 
    component

    sds: length k array
    "sd" parameter of SGG distribution for each component

    p_stars: length k array
    "p_star" parameters (can be transformed to "p") of SGG distribution for
    each component

    q_stars: length k array
    "q_star" parameters (can be transformed to "q") of SGG distribution for
    each component

    expon:
    exponent for scaling Temperatures

    Returns
    -------
    pdfs: 1000 x k array
    SGG probability density function

    scaled_pdfs: 1000 x k array
    Proportion of magnetization unblocking at a 

    pred_data: 1000 x 3 array
    Predicted interpolated Zijderveld data 
    """
    ps=jnp.exp(p_stars)
    qs=2/jnp.pi*jnp.arctan(1/q_stars)
    Ts_scaled=jnp.linspace(jnp.min(Ts_data),jnp.max(Ts_data),1000)
    Ts_interp=unscale_xs(Ts_scaled,expon)
    pdfs=jnp.empty((len(cs),len(Ts_interp)))
    pred_data=construct_pred_data(Bs,Ts_scaled,cs,mus,sds,ps,qs)
    exponexpon=jnp.exp(-jnp.abs(expon))
    for i in range(len(cs)):
        pdfs=pdfs.at[i].set(cs[i]*SGG_pdf(Ts_scaled,mus[i],sds[i],ps[i],qs[i])*exponexpon*(np.max(Ts_interp)*jnp.heaviside(expon,0)-Ts_interp*(jnp.heaviside(expon,0)*2-1))**(exponexpon-1))
    pdfs=jnp.nan_to_num(pdfs,0)
    scaled_pdfs=pdfs/jnp.sum(pdfs,axis=0)
    return(pdfs,scaled_pdfs,pred_data)

def plot_B_ratios(ax,zijd_data,Bs,Ts_data,cs,mus,sds,p_stars,q_stars,expon,alpha=1,anchored=False,shared_axis=0,**kwargs):
    """
    Plot interpolated demag data onto Zijderveld plot axes. Plots ratio of 
    components unblocking at a particular temperature as a colormap. The hue of
    the colormap represents the dominant component, whereas the brightness or
    value represents the degree of overlap, with white colors representing 
    overlapping components, and darker colors representing each component.

    Inputs
    ------
    ax: matplotlib axis
    axis on which to plot the data

    zijd_data: n x 3 array
    Set of demagnetization data.

    Bs: k x 3 array
    Field directions

    Ts_data: length n array
    Scaled temperatures/coercivities of actual demag data

    cs: length K array
    Magnitudes of each component

    mus: length k array
    "mu" parameter of SGG distribution for each 
    component

    sds: length k array
    "sd" parameter of SGG distribution for each component

    p_stars: length k array
    "p_star" parameters (can be transformed to "p") of SGG distribution for
    each component

    q_stars: length k array
    "q_star" parameters (can be transformed to "q") of SGG distribution for
    each component

    alpha: float
    alpha of plotted data

    anchored: bool
    When True, performs the TROUT fit to all data. TROUT assumes that at high
    temperatures, the magnetization goes to zero. With Anchored=False, the 
    origin is moved to the final demagnetization step so this requirement is
    not met. With Anchored=True an additional component may be needed for 
    specimens where the magnetization does not trend towards the origin.

    shared_axis: int
    Index of axis (x, y or z) is shared between both zijderveld projections.
    
    Returns
    -------
    scaled_pdfs: 1000 x k array
    proportions of SGG pdfs.
    """
    pdfs,scaled_pdfs,pred_data=get_B_ratios(Bs,Ts_data,cs,mus,sds,p_stars,q_stars,expon)
    
    pdfs=pdfs[:,:-1]
    scaled_pdfs=scaled_pdfs[:,:-1]
    scaled_pdfs=scaled_pdfs[:,np.amax(pdfs,axis=0)>0]
    pred_data=pred_data[:-1]
    pred_data=pred_data[np.amax(pdfs,axis=0)>0]
    pdfs=pdfs[:,np.amax(pdfs,axis=0)>0]
    norm=plt.Normalize(0,1)
    if anchored==False:
        pred_data+=zijd_data[-1]
    axes=np.arange(3)
    unshared=axes[axes!=shared_axis]
    
    lines1=np.array([[pred_data[:-1,shared_axis],pred_data[1:,shared_axis]],[pred_data[:-1,unshared[0]],pred_data[1:,unshared[0]]]]).T
    lines2=np.array([[pred_data[:-1,shared_axis],pred_data[1:,shared_axis]],[pred_data[:-1,unshared[1]],pred_data[1:,unshared[1]]]]).T
    if len(pdfs)==2:
        lc=LineCollection(lines1,cmap='RdBu',norm=norm,**kwargs)
        lc2=LineCollection(lines2,cmap='RdBu',norm=norm,**kwargs)
        colors=scaled_pdfs[0]
        colors=colors.at[np.isnan(colors)==True].set(0)
        lc.set_array(colors)
        lc2.set_array(colors)
    else:
        poss_hs=np.linspace(0,1,len(pdfs)+1)[1:]
        hs=poss_hs[np.argmax(scaled_pdfs,axis=0)]
        maxfracs=np.amax(scaled_pdfs,axis=0)
        maxfracs=maxfracs.at[maxfracs<0.5].set(0.5)
        sats=np.abs(maxfracs-0.5)*2
        vs=np.exp(-(maxfracs-0.5)**2/0.3)
        hsv=np.array([hs,sats,vs]).T
        rgb=hsv_to_rgb(hsv)
        lc=LineCollection(lines1,colors=rgb,**kwargs)
        lc2=LineCollection(lines2,colors=rgb,**kwargs)

        


    lc.set_alpha(alpha)
    lc2.set_alpha(alpha)
    ax.add_collection(lc)
    ax.add_collection(lc2)
    return(scaled_pdfs)

def plot_net(ax=None):
    """
    Draws circle and tick marks for equal area projection.
    (From PmagPy)

    Inputs
    ------
    ax: matplotlib axis

    Returns
    -------
    None
    """
    if ax==None:
        ax=plt.gca()
    ax.axis("off")
    Dcirc = np.arange(0, 361.)
    Icirc = np.zeros(361, 'f')
    Xcirc, Ycirc = [], []
    for k in range(361):
        XY = pmag.dimap(Dcirc[k], Icirc[k])
        Xcirc.append(XY[0])
        Ycirc.append(XY[1])
    ax.plot(Xcirc, Ycirc, 'k')

# put on the tick marks
    Xsym, Ysym = [], []
    for I in range(10, 100, 10):
        XY = pmag.dimap(0., I)
        Xsym.append(XY[0])
        Ysym.append(XY[1])
    ax.plot(Xsym, Ysym, 'k+')
    Xsym, Ysym = [], []
    for I in range(10, 90, 10):
        XY = pmag.dimap(90., I)
        Xsym.append(XY[0])
        Ysym.append(XY[1])
    ax.plot(Xsym, Ysym, 'k+')
    Xsym, Ysym = [], []
    for I in range(10, 90, 10):
        XY = pmag.dimap(180., I)
        Xsym.append(XY[0])
        Ysym.append(XY[1])
    ax.plot(Xsym, Ysym, 'k+')
    Xsym, Ysym = [], []
    for I in range(10, 90, 10):
        XY = pmag.dimap(270., I)
        Xsym.append(XY[0])
        Ysym.append(XY[1])
    ax.plot(Xsym, Ysym, 'k+')
    for D in range(0, 360, 10):
        Xtick, Ytick = [], []
        for I in range(4):
            XY = pmag.dimap(D, I)
            Xtick.append(XY[0])
            Ytick.append(XY[1])
        ax.plot(Xtick, Ytick, 'k')
    ax.axis("equal")
    ax.axis((-1.05, 1.05, -1.05, 1.05))

def plot_di(dec=None, inc=None, di_block=None, color='k', marker='o', markersize=20, legend='no', label='', title=None, edge=None,alpha=1,ax=None):
    """
    Modified from pmagpy.

    Plot declination, inclination data on an equal area plot.

    Before this function is called a plot needs to be initialized with code 
    that looks something like:
    >fignum = 1
    >plt.figure(num=fignum,figsize=(10,10),dpi=160)
    >ipmag.plot_net(fignum)

    Required Parameters
    -----------
    dec : declination being plotted
    inc : inclination being plotted

    or

    di_block: a nested list of [dec,inc,1.0]
    (di_block can be provided instead of dec, inc in which case it will be 
    used)

    Optional Parameters (defaults are used if not specified)
    -----------
    color : the default color is black. Other colors can be chosen (e.g. 'r')
    marker : the default marker is a circle ('o')
    markersize : default size is 20
    label : the default label is blank ('')
    legend : the default is no legend ('no'). Putting 'yes' will plot a legend.
    edge : marker edge color - if blank, is color of marker
    alpha : opacity
    """
    if ax==None:
        ax=plt.gca()
    X_down = []
    X_up = []
    Y_down = []
    Y_up = []
    color_down = []
    color_up = []

    if di_block is not None:
        di_lists = ipmag.unpack_di_block(di_block)
        if len(di_lists) == 3:
            dec, inc, intensity = di_lists
        if len(di_lists) == 2:
            dec, inc = di_lists
    try:
        length = len(dec)
        for n in range(len(dec)):
            XY = pmag.dimap(dec[n], inc[n])
            if inc[n] >= 0:
                X_down.append(XY[0])
                Y_down.append(XY[1])
                if type(color) == list:
                    color_down.append(color[n])
                else:
                    color_down.append(color)
            else:
                X_up.append(XY[0])
                Y_up.append(XY[1])
                if type(color) == list:
                    color_up.append(color[n])
                else:
                    color_up.append(color)
    except:
        XY = pmag.dimap(dec, inc)
        if inc >= 0:
            X_down.append(XY[0])
            Y_down.append(XY[1])
            color_down.append(color)
        else:
            X_up.append(XY[0])
            Y_up.append(XY[1])
            color_up.append(color)

    if len(X_up) > 0:
        ax.scatter(X_up, Y_up, facecolors='none', edgecolors=color_up,
                    s=markersize, marker=marker, label=label,alpha=alpha)

    if len(X_down) > 0:
        ax.scatter(X_down, Y_down, facecolors=color_down, edgecolors=edge,
                    s=markersize, marker=marker, label=label,alpha=alpha)
    if legend == 'yes':
        ax.legend(loc=2)
    if title != None:
        ax.set_title(title)

def dist_plot(ax,Ts_base,cs,mus,sds,p_stars,q_stars,expon,colors,fill=False,**kwargs):
    """
    Plots SGG distributions for each component in the TROuT model.

    Inputs
    ------
    Ts_base: length n array
    Temperatures/coercivities of demag data

    cs: length K array
    Magnitudes of each component

    mus: length k array
    "mu" parameter of SGG distribution for each 
    component

    sds: length k array
    "sd" parameter of SGG distribution for each component

    p_stars: length k array
    "p_star" parameters (can be transformed to "p") of SGG distribution for
    each component

    q_stars: length k array
    "q_star" parameters (can be transformed to "q") of SGG distribution for
    each component

    expon: float
    exponent to scale data by. If expon > 0 then
    data are scaled by their distance from the 
    highest temperature/field.

    colors: length k array
    colors to plot each component
    
    fill: bool
    Whether to shade distributions or just plot lines.
    """
    ps=np.exp(p_stars)
    qs=2/np.pi*np.arctan(1/q_stars)
    Ts_interp=np.linspace(min(Ts_base),max(Ts_base),1000)
    Ts_scaled=scale_xs(Ts_interp,expon)
    exponexpon=np.exp(-np.abs(expon))
    for i in range(len(mus)):
        pdf=cs[i]*SGG_pdf(Ts_scaled,mus[i],sds[i],ps[i],qs[i])
        pdf=pdf*exponexpon*(np.max(Ts_interp)*jnp.heaviside(expon,0)-Ts_interp*(jnp.heaviside(expon,0)*2-1))**(exponexpon-1)
        pdf=jnp.nan_to_num(pdf,0)
        ax.plot(Ts_interp[1:-1],pdf[1:-1],color=colors[i],**kwargs)
        if fill==True:
            ax.fill_between(Ts_interp[1:-1],pdf[1:-1],color=colors[i],**kwargs)
            
def plot_results(zijd_data,Ts_base,Ts_data,par_dict,expon,anchored=False,shared_axis=0,ax=None,hide_xlabel=False,inset_loc='upper left'):
    """
    Plots TROUT model on Zijderveld plot, SGG distributions and equal area plot.
    
    Inputs
    ------
    zijd_data: nx3 array
    Set of Zijderveld data for 

    Ts_base: length n array
    Original Temperature steps (or coercivities).

    Ts_data: length n array
    Scaled Temperature steps or coercivities

    par_dict: dict
    Dictionary of TROUT parameters
    
    expon: float
    exponent to scale data by. If expon > 0 then
    data are scaled by their distance from the 
    highest temperature/field.

    anchored: bool
    When True, performs the TROUT fit to all data. TROUT assumes that at high
    temperatures, the magnetization goes to zero. With Anchored=False, the 
    origin is moved to the final demagnetization step so this requirement is
    not met. With Anchored=True an additional component may be needed for 
    specimens where the magnetization does not trend towards the origin.

    shared_axis: int
    Index of axis (x, y or z) is shared between both zijderveld projections.

    ax: list of matplotlib axes or None
    Axes to plot data to. If None plots on a new figure.

    hide_xlabel: bool
    If True hides x label (for multi-column subplots)

    inset_loc: str
    Location of inset Equal Area plot.

    """    
    n_components=len(par_dict['Bs'])
    if np.all(ax)==None:
        fig,ax=plt.subplots(1,2,figsize=(12,4));
    plot_zijd_data(ax[0],zijd_data,Ts_base,shared_axis=shared_axis)
    
    zijd_dirs=pmag.cart2dir(zijd_data)
    


    Bs=np.array(par_dict['Bs'])
    cs=np.array(par_dict['cs'])
    mus=np.array(par_dict['mus'])
    sds=np.array(par_dict['sds'])
    p_stars=np.array(par_dict['p_stars'])
    q_stars=np.array(par_dict['q_stars'])

    Ts_interp=np.linspace(0,max(Ts_base),1000)
    plot_B_ratios(ax[0],zijd_data,Bs,Ts_data,cs,mus,sds,p_stars,q_stars,expon,lw=4,label='Naive Guess',anchored=anchored,shared_axis=shared_axis);
    
    hs=np.linspace(0,1,n_components+1)[1:]
    sats=np.ones(n_components)
    vs=np.full(n_components,0.8)
    hsvs=np.array([hs,sats,vs]).T
    colors=hsv_to_rgb(hsvs)
    if n_components==2:
        colors=['b','r']
    dist_plot(ax[1],Ts_base,cs,mus,sds,p_stars,q_stars,expon,colors)
    
    axes_names=['x','y','z']
    if hide_xlabel==False:
        ax[0].set_xlabel(axes_names[shared_axis]+', arbitrary units')
        ax[1].set_xlabel('T (°C)')
    axes_names.remove(axes_names[shared_axis])
    
    ax[0].set_ylabel(axes_names[0]+'/'+axes_names[1]+', arbitrary units')
    ax[1].set_ylabel('f(T) (Scaled)');
   
    
    axins = inset_axes(ax[1], width='65%', height='65%',loc=inset_loc)
    plot_net(ax=axins)
    plot_di(zijd_dirs[:,0],zijd_dirs[:,1],ax=axins)
    B_dirs=pmag.cart2dir(Bs)
    if len(B_dirs)>1:
        xs,ys=pmag.dimap(B_dirs[:,0],B_dirs[:,1]).T

        for i in range(n_components):
            if B_dirs[i,1]>0:
                facecolor=colors[i]
            else:
                facecolor=None
                
            axins.plot(xs[i],ys[i],'s',markeredgecolor=colors[i],markerfacecolor=facecolor,markersize=5)
    else:
        xs,ys=pmag.dimap(B_dirs[:,0],B_dirs[:,1])
        if B_dirs[0,1]>0:
            facecolor=colors[0]
        else:
            facecolor=None
        axins.plot(xs,ys,'s',markeredgecolor=colors[0],markerfacecolor=facecolor,markersize=5)
    plt.tight_layout()
            
def gen_initialization_points(pars,bounds,init_scale,pop_size):
    """
    Generates initialization points for a minimization from a set of best
    guess parameters given their bounds using a truncated normal distribution

    Inputs
    ------
    pars:
    Vector of best guess parameters

    bounds:
    Bounds on best guess parameters

    pop_size:
    Number of initializations (as a function of number of parameters). E.g.
    with 16 parameters and pop_size=2, generates 16 points
    """
    init=np.empty((pop_size*len(pars),len(pars)))
    for i in range(len(pars)):
        init[:,i]=truncnorm.rvs(loc=pars[i],scale=init_scale,a=(bounds[i][0]-pars[i])/init_scale,b=(bounds[i][1]-pars[i])/init_scale,size=pop_size*len(pars))
    return(init)

def create_scales_and_bounds(n_components,Trange):
    """
    !!DEPRECATED!!

    Creates scales and bounds for a TROUT model with a particular number of 
    components. When minimizing, parameters are scaled such that they have
    bounds of (0,2) or (1,-1)

    Inputs
    ------
    n_components: int
    Number of components (k)

    Trange: 
    Total range of temperature steps
    """
    scales=[]
    bounds=[]
    scale_nums=[2,Trange/2,Trange*5/6,2.5,5]
    bound_nums=[[-1,1],[0,2],[0,2],[0,2],[-1,1]]
    for j in range(len(scale_nums)):
        for i in range(n_components):
            scale=scale_nums[j]
            bound=bound_nums[j]
            if j==0:
                for k in range(3):
                    scales.append(scale)
                    bounds.append(bound)
            else:
                scales.append(scale)
                bounds.append(bound)
    scales.append(1)
    scales.append(np.radians(5))
    bounds.append([0,1])
    bounds.append([0,2])
    return(np.array(scales)/1e2,np.array(bounds)*1e2)
    

def create_minimizer_vecs(Ts_data,Bs,cs,mus,sds,p_stars,q_stars,post_func):
    """
    Taking a set of parameters from the TROUT model, finds numbers to scale the
    parameters by so that the prior bounds are (0,2) or (-1,1) and generates 
    these bounds. Converts the parameters, scales and bounds into a vector.

    Inputs
    ------
    Ts_data: length n array
    Temperatures at which demag data is evaluated. (This is included in the 
    prior only to set a range for the acceptable bounds of mu)

    Bs: length k array
    Field directions for each component of demag data

    cs: length K array
    Magnitudes of each component

    mus: length k array
    "mu" parameter of SGG distribution for each 
    component

    sds: length k array
    "sd" parameter of SGG distribution for each component

    p_stars: length k array
    "p_star" parameters (can be transformed to "p") of SGG distribution for
    each component

    q_stars: length k array
    "q_star" parameters (can be transformed to "q") of SGG distribution for
    each component

    post_func: function
    Posterior function

    Returns
    -------
    pars: array
    1d vector of parameters

    scales: array
    1d vector of scales

    bounds: array
    array of parameter bounds.
    """
    #Combine Bs and cs into "Ms" parameter
    Ms=np.transpose(Bs.T*cs)
    #Create dictionaries of parameters, scales for parameters, upper and lower bounds
    
    par_dict={'Ms':Ms,
              'mus':mus.astype('float32'),
              'sds':sds,
              'p_stars':p_stars,
              'q_stars':q_stars}
    
    scale_dict={'Ms':np.full(Ms.shape,2.),
                'mus':np.full(mus.shape,(np.max(Ts_data)-np.min(Ts_data))/2.),
                'sds':np.full(sds.shape,(np.max(Ts_data)-np.min(Ts_data))*10/6),
                'p_stars':np.full(p_stars.shape,2.5),
                'q_stars':np.full(q_stars.shape,5.)}
    
    bounds_lower_dict={'Ms':np.full(Ms.shape,-1.),
                       'mus':np.full(mus.shape,0.),
                       'sds':np.full(mus.shape,0.),
                       'p_stars':np.full(p_stars.shape,0.),
                       'q_stars':np.full(q_stars.shape,-1.)}
    
    bounds_upper_dict={'Ms':np.full(Ms.shape,1.),
                       'mus':np.full(mus.shape,2.),
                       'sds':np.full(sds.shape,2.),
                       'p_stars':np.full(p_stars.shape,2.),
                       'q_stars':np.full(q_stars.shape,1.)}
    
    #If using multivariate gaussian noise with misorientation noise, add "sigma" and "psi" terms
    if post_func in [post_func_sigma_psi,post_func_grad]:
        par_dict['sigma']=1e-1
        scale_dict['sigma']=1
        bounds_lower_dict['sigma']=0
        bounds_upper_dict['sigma']=1
        par_dict['psi']=np.radians(2)
        scale_dict['psi']=np.radians(5)
        bounds_lower_dict['psi']=0
        bounds_upper_dict['psi']=2
    
    #Convert pars, scales bounds to vectors
    pars,nothing=ravel_pytree(par_dict)
    scales,nothing=ravel_pytree(scale_dict)
    bounds_lower,nothing=ravel_pytree(bounds_lower_dict)
    bounds_upper,nothing=ravel_pytree(bounds_upper_dict)
    bounds=np.array([bounds_lower,bounds_upper]).T
    
    #Scale pars, scales and bounds
    bounds*=100
    scales/=100
    pars/=scales
    return(pars,scales,bounds)

    
def find_best_fit_model(zijd_data,Ts_data,n_components,anchored=False,init_scale=0.1,pop_size=2,polish=True,post_func=post_func_sigma_psi,use_grad=False,bpoints=None):
    """
    Finds the best fitting model to the data using the BFGS method.

    zijd_data: nx3 array
    Set of demagnetization data the TROUT model will be applied to

    Ts_data: length n array
    Scaled Temperature steps or coercivities
    
    n_components: int
    number of components (k) in model.

    anchored: bool
    When True, performs the TROUT fit to all data. TROUT assumes that at high
    temperatures, the magnetization goes to zero. With Anchored=False, the 
    origin is moved to the final demagnetization step so this requirement is
    not met. With Anchored=True an additional component may be needed for 
    specimens where the magnetization does not trend towards the origin.

    init_scale: float
    How much of the prior space to sample around the best guess. Larger numbers
    may find more solutions around the best guess but may have significantly
    more failed results.

    pop_size: int
    Number of initializations (as a function of number of parameters). E.g.
    with 16 parameters and pop_size=2, generates 16 points

    polish: bool
    If True performs a second minimization on the best result to "polish" it
    off.

    post_func: function
    Function of posterior distribution being evaluated

    use_grad: bool
    If True fits to the gradient of the data (not advised).

    bpoints: list
    list of temperatures to split the magnetization at for an initial guess,
    should probably not be used unless the initial guess is unfeasible.

    Returns
    -------
    par_dict: dict
    Dictionary of parameters of best fitting TROUT result

    alt_results: list of dicts
    Alternative results from the TROUT model with lower posterior probabilities
    """
    #Subtract the last data point in the case of an unanchored fit (don't force through origin)
    if anchored==False:
        zijd_data=zijd_data-zijd_data[-1]
    
    #Generate negative log posterior and function to "unravel" posterior distribution
    unravel_func,nlp=create_unravel_funcs(n_components,post_func)
    
    #Use "Naive" guess partitioning data into n_components pieces
    Bs,cs,mus,sds,p_stars,q_stars=find_naive_fit(zijd_data,Ts_data,n_components,bpoints=bpoints,anchored=anchored)
    
    #Create vectors for minimizer
    pars,scales,bounds=create_minimizer_vecs(Ts_data,Bs,cs,mus,sds,p_stars,q_stars,post_func)
        
    #Generate points for initialization of data within bounds
    init_points=gen_initialization_points(pars,bounds,init_scale*100,pop_size)
    
    #Total length of points (for prior)
    vds=jnp.sum(jnp.linalg.norm(jnp.diff(zijd_data,axis=0),axis=1))+jnp.linalg.norm(zijd_data[-1])
    if post_func==post_func_grad:
        zijd_diffs=-np.gradient(zijd_data,Ts_data/np.ptp(Ts_data),axis=0)
    
    if use_grad:
        gradient=jit(grad(nlp))
    else:
        gradient=None
    #Use BFGS minimization to produce results
    results=[]
    for x0 in init_points:
        if post_func==post_func_grad:
            result=minimize(nlp,
                            x0,
                            args=(zijd_data,zijd_diffs,Ts_data,scales),
                            method='BFGS',
                            bounds=bounds,
                            options={'gtol':1e-1*len(zijd_data)/37,
                                'eps':np.sqrt(np.finfo('float32').eps)})
        else:
            result=minimize(nlp,
                            x0,
                            args=(zijd_data,Ts_data,scales),
                            method='BFGS',
                            bounds=bounds,
                            jac=gradient,
                            options={'gtol':1*len(zijd_data)/37,
                                'eps':np.sqrt(np.finfo('float32').eps)})
        
        if result.success:
            results.append(result)
            
    #Calculate result with highest posterior probability
    maxresult=None
    minfun=np.inf
    for result in results:
        fun=result.fun        
        if (minfun>fun)&(result.success==True):
            minfun=fun
            maxresult=result
    
    
    #If "polish" argument is set (on by default), perform an additional 
    #non gradient-based minimization on the best performing model
    if polish==True:
        if post_func==post_func_grad:
            result=minimize(nlp,
                            x0,
                            args=(zijd_data,zijd_diffs,Ts_data,scales),
                            method='Nelder-Mead',
                            bounds=np.array(bounds))
        else:
            maxresult=minimize(nlp,
                               maxresult.x,
                               args=(zijd_data,Ts_data,scales),
                               method='Nelder-Mead',
                               bounds=np.array(bounds))

    
    
    #Generate parameter dictionary for returning
    par_dict=unravel_func(maxresult.x*scales)
    par_dict['Bs']=(par_dict['Ms'].T/np.linalg.norm(par_dict['Ms'],axis=1)).T
    par_dict['cs']=np.linalg.norm(par_dict['Ms'],axis=1)
    par_dict['logp']=-maxresult.fun
    
    #Generate parameter dictionaries for alternate results
    alt_results=[]
    for result in results:
        alt_par_dict=unravel_func(result.x*scales)
        alt_par_dict['Bs']=(alt_par_dict['Ms'].T/np.linalg.norm(alt_par_dict['Ms'],axis=1)).T
        alt_par_dict['cs']=np.linalg.norm(alt_par_dict['Ms'],axis=1)
        alt_par_dict['logp']=-result.fun
        alt_results.append(alt_par_dict)
    print(str(len(results))+' of '+str(len(init_points))+' minimizations successful')
    print('Best result has a log probability score of %3.1f'%par_dict['logp'])
    return(par_dict,alt_results)

def find_best_fit_emcee(zijd_data,Ts_data,n_components,n_samples,anchored=False,init_scale=0.1,pop_size=2,post_func=post_func_sigma_psi,bpoints=None):
    """
    Finds the best distribution of models which can fit the data well using the
    ensemble Markov Chain Monte Carlo Sampler (emcee).

    Inputs
    ------
    zijd_data: nx3 array
    Set of demagnetization data the TROUT model will be applied to

    Ts_data: length n array
    Scaled Temperature steps or coercivities
    
    n_components: int
    number of components (k) in model.

    n_samples: int
    number of samples in the MCMC chain

    anchored: bool
    When True, performs the TROUT fit to all data. TROUT assumes that at high
    temperatures, the magnetization goes to zero. With Anchored=False, the 
    origin is moved to the final demagnetization step so this requirement is
    not met. With Anchored=True an additional component may be needed for 
    specimens where the magnetization does not trend towards the origin.

    init_scale: float
    How much of the prior space to sample around the best guess. Larger numbers
    may find more solutions around the best guess but may have significantly
    more failed results.

    pop_size: int
    Number of initializations (as a function of number of parameters). E.g.
    with 16 parameters and pop_size=2, generates 16 points

    polish: bool
    If True performs a second minimization on the best result to "polish" it
    off.

    post_func: function
    Function of posterior distribution being evaluated

    use_grad: bool
    If True fits to the gradient of the data (not advised).

    bpoints: list
    list of temperatures to split the magnetization at for an initial guess,
    should probably not be used unless the initial guess is unfeasible.

    Returns
    -------
    par_dict: dict
    Dictionary of parameters of median result from MCMC sample.

    par_dicts: list of dicts
    Sample of 100 results from the MCMC chain.
    """
    #Subtract the last data point in the case of an unanchored fit (don't force through origin)
    if anchored==False:
        zijd_data=zijd_data-zijd_data[-1]
    
    #Generate negative log posterior and function to "unravel" posterior distribution
    unravel_func,nlp=create_unravel_funcs(n_components,post_func)
    
    #Use "Naive" guess partitioning data into n_components pieces
    Bs,cs,mus,sds,p_stars,q_stars=find_naive_fit(zijd_data,Ts_data,n_components,bpoints=bpoints,anchored=anchored)
    
    #Create vectors for minimizer
    pars,scales,bounds=create_minimizer_vecs(Ts_data,Bs,cs,mus,sds,p_stars,q_stars,post_func)
        
    #Generate points for initialization of data within bounds
    init_points=gen_initialization_points(pars,bounds,init_scale*100,pop_size)
    
    #Total length of points (for prior)
    vds=jnp.sum(np.linalg.norm(jnp.diff(zijd_data,axis=0),axis=1))+jnp.linalg.norm(zijd_data[-1])
 
    @jit 
    def lp(pars,zijd_data,Ts_data,scales):
        return(-nlp(pars,zijd_data,Ts_data,scales))
    
    for i in range(len(init_points)):
        while lp(init_points[i],zijd_data,Ts_data,scales) < -1e+30:
            new_init=gen_initialization_points(pars,bounds,init_scale*100,pop_size)
            init_points[i]=new_init[i]
            
    sampler = emcee.EnsembleSampler(
        len(init_points), len(pars), lp, args=(zijd_data,Ts_data,scales))
    try:
        sampler.run_mcmc(init_points,n_samples,progress=True)
    except:

        plt.plot(sampler.get_log_prob())
    samples=sampler.get_chain()
    samples=samples[int(n_samples/2):]
    samples=samples.reshape(samples.shape[0]*samples.shape[1],len(pars))
    samples*=scales
    median_sample=np.median(samples,axis=0)
   
    par_dict= unravel_func((median_sample).astype('float32'))
    par_dict['Bs']=(par_dict['Ms'].T/np.linalg.norm(par_dict['Ms'],axis=1)).T
    par_dict['cs']=np.linalg.norm(par_dict['Ms'],axis=1)
    cs=np.random.choice(range(len(samples)),100)
    thinned_samples=samples[cs]
    par_dicts=[]
    for sample in thinned_samples:
        par_dict_new= unravel_func((sample).astype('float32'))
        par_dict_new['Bs']=(par_dict_new['Ms'].T/np.linalg.norm(par_dict_new['Ms'],axis=1)).T
        par_dict_new['cs']=np.linalg.norm(par_dict_new['Ms'],axis=1)
        par_dicts.append(par_dict_new)
    
    return(par_dict,par_dicts)
    

def plot_di_hist(zijd_data,carts,bins=100,site_dir=[[-1000,-1000]],levels=30):
    """
    Plots a distribution of directional results as a contour plot. 
    
    Inputs
    ------
    zijd_data: n x 3 array
    Demagnetization data for a spectimen

    carts: n_samples x 3 array
    cartesian directions of data

    bins: int
    number of bins for histogram of densities

    site_dir: length 2 array
    Array of [Declination, Inclination] of expected site direction

    levels: int
    number of levels for contour plot.

    Returns
    -------
    None
    """
    zijd_dir=pmag.cart2dir(zijd_data)
    if carts.shape[1]==2:
        cmaps=['Blues','Reds']
        labels=['Low Temperature Component','High Temperature Component']
    else:
        cmaps=['Blues','Greens','Reds','Greys','Viridis','Inferno']
        labels=['Low Temperature Component','Intermediate Temperature Component','High Temperature Component','Higher Temperature Component']
    for i in range(carts.shape[1]):
        dirs=pmag.cart2dir(carts[:,i])
        xs,ys=pmag.dimap(dirs[:,0], dirs[:,1]).T
        xedges=np.linspace(-1,1,bins)
        yedges=np.linspace(-1,1,bins)
        counts,x,y=np.histogram2d(xs,ys,bins=np.array([xedges,yedges]))
        xcenters=np.diff(xedges)+xedges[:-1]
        ycenters=np.diff(yedges)+yedges[:-1]
        ipmag.plot_net()
        contours=plt.contourf(xcenters,ycenters, counts.T, cmap=cmaps[i],extend='both',label=labels[i],vmin=1,levels=levels)
        cmap=contours.get_cmap()
        cmap.set_under([0,0,0,0])
        contours.cmap.set_under([0,0,0,0])
    ipmag.plot_di(zijd_dir[:,0],zijd_dir[:,1],label='Zijderveld Data',alpha=0.5)
    if site_dir[0,0]!=-1000:
        for i in site_dir:
            ipmag.plot_di(i[0],i[1],marker='*',markersize=100,color='cyan')

def plot_results_emcee(zijd_data,Ts_base,Ts_data,alt_dicts,expon,anchored=False,shared_axis=0,ax=None,hide_xlabel=False,inset_loc='upper left'):
    """
    Plots results of a fit using the markov chain monte carlo sampler.
    Syntax is the same as plot_results.
    """    
    if np.all(ax)==None:
        fig,ax=plt.subplots(1,2,figsize=(12,4));
    plot_zijd_data(ax[0],zijd_data,Ts_base,shared_axis=shared_axis)
    axins = inset_axes(ax[1], width='65%', height='65%',loc=inset_loc)
    zijd_dirs=pmag.cart2dir(zijd_data)
    plot_net(ax=axins)
    n_components=len(alt_dicts[0]['Bs'])
    xarray=np.empty((100,n_components))
    
    yarray=np.empty((100,n_components))
    i=0
    for alt_dict in np.random.choice(alt_dicts,100):
        n_components=len(alt_dict['Bs'])

        Bs=np.array(alt_dict['Bs'])
        cs=np.array(alt_dict['cs'])
        mus=np.array(alt_dict['mus'])
        sds=np.array(alt_dict['sds'])
        p_stars=np.array(alt_dict['p_stars'])
        q_stars=np.array(alt_dict['q_stars'])
        Ts_interp=np.linspace(0,max(Ts_base),1000)
        plot_B_ratios(ax[0],zijd_data,Bs,Ts_data,cs,mus,sds,p_stars,q_stars,expon,lw=4,label='Naive Guess',anchored=anchored,shared_axis=shared_axis,alpha=0.05);
    
        hs=np.linspace(0,1,n_components+1)[1:]
        sats=np.ones(n_components)
        vs=np.full(n_components,0.8)
        hsvs=np.array([hs,sats,vs]).T
        colors=hsv_to_rgb(hsvs)
        if n_components==2:
            colors=['b','r']
        dist_plot(ax[1],Ts_base,cs,mus,sds,p_stars,q_stars,expon,colors,alpha=0.1)
    
        axes_names=['x','y','z']
        if hide_xlabel==False:
            ax[0].set_xlabel(axes_names[shared_axis]+', arbitrary units')
            ax[1].set_xlabel('T (°C)')
        axes_names.remove(axes_names[shared_axis])
    
        ax[0].set_ylabel(axes_names[0]+'/'+axes_names[1]+', arbitrary units')
        ax[1].set_ylabel('f(T) (Scaled)');
        plot_net(ax=axins)
        plot_di(zijd_dirs[:,0],zijd_dirs[:,1],ax=axins)
        B_dirs=pmag.cart2dir(Bs)
        if len(B_dirs)>1:
            xs,ys=pmag.dimap(B_dirs[:,0],B_dirs[:,1]).T

            for i in range(n_components):
                if B_dirs[i,1]>0:
                    facecolor=colors[i]
                else:
                    facecolor=None

                axins.plot(xs[i],ys[i],'s',color=colors[i],markersize=5,alpha=0.1)
        else:
            xs,ys=pmag.dimap(B_dirs[:,0],B_dirs[:,1])
            if B_dirs[0,1]>0:
                facecolor=colors[0]
            else:
                facecolor=None
            axins.plot(xs,ys,'s',color=colors[0],markerfacecolor=facecolor,markersize=5)
    

        B_dirs=pmag.cart2dir(Bs)
    
def TROUT_write(WD,specimen,n_components,**kwargs):
    """
    Runs the TROUT model and writes result for a specimen to the MagIC format
    to specimen table.
    
    Inputs:
    -------
    WD: str
    Working directory for specimen table
    
    specimen: str
    name of specimen
    
    n_components: int
    number of components fitted using TROUT model
    
    Returns
    -------
    None (writes to tables directly).
    """
    print('Writing data for specimen '+specimen)
    meas_data,sample_data,site_data=import_direction_data(WD)
    zijd_data,Ts_base,Ts_data,Ts,datatype,expon=prepare_specimen_data(specimen,meas_data,sample_data,site_data)
    par_dict,alt_par_dicts=find_best_fit_model(zijd_data,Ts_data,n_components,**kwargs)
    overlaps,crossovers,lowers,uppers=calc_overlap_range(par_dict,Ts_base,expon)
    specimens=pd.read_csv(WD+'specimens.txt',sep='\t',skiprows=1)

    B_dirs=pmag.cart2dir(par_dict['Bs'])
    for column in ['dir_dec','dir_inc','meas_step_min','meas_step_max','dir_n_comps','dir_comp','meas_step_unit']:
        if column not in specimens.columns:
            specimens[column]=np.nan
    if 'method_codes' not in specimens.columns:
        specimens['method_codes']=''
        
    for k in range(n_components):
        fit='Fit '+str(n_components-k)
        
        TROUTfilter=np.any(specimens.loc[specimens.specimen==specimen,'method_codes'].str.contains('DE-TROUT'))
        fitfilter=len(specimens.loc[(specimens.specimen==specimen)&(specimens.dir_comp==fit)])>=1
        if TROUTfilter and fitfilter:
            
            specimens.loc[(specimens.specimen==specimen)&(specimens.dir_comp==fit),'dir_dec']=B_dirs[k,0]
            specimens.loc[(specimens.specimen==specimen)&(specimens.dir_comp==fit),'dir_inc']=B_dirs[k,1]
            specimens.loc[(specimens.specimen==specimen)&(specimens.dir_comp==fit),'meas_step_min']=lowers[k,k]+273
            specimens.loc[(specimens.specimen==specimen)&(specimens.dir_comp==fit),'meas_step_max']=uppers[k,k]+273
            specimens.loc[(specimens.specimen==specimen)&(specimens.dir_comp==fit),'meas_step_max']=uppers[k,k]+273
            specimens.loc[(specimens.specimen==specimen)&(specimens.dir_comp==fit),'dir_n_comps']=n_components
            specimens.loc[(specimens.specimen==specimen)&(specimens.dir_comp==fit),'dir_comp']=fit
            specimens.loc[(specimens.specimen==specimen)&(specimens.dir_comp==fit),'meas_step_unit']='K'

        else:
            sample=specimens.loc[specimens.specimen==specimen,'sample'].iloc[0]
            specimens=specimens.append({'specimen':specimen,'sample':sample,'dir_dec':B_dirs[k,0],
                                        'dir_inc':B_dirs[k,1],'dir_n_comps':n_components,
                                        'dir_comp':fit,'meas_step_min':lowers[k,k]+273,
                                        'meas_step_max':uppers[k,k]+273,'method_codes':'LP-DIR-T:DE-TROUT',
                                        'meas_step_unit':'K'},ignore_index=True)
    pmag.magic_write(WD+'specimens.txt',specimens,'specimens',dataframe=True)