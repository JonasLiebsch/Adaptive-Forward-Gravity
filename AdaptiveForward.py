# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 08:27:17 2021

@author: Jonas Liebsch
liebschjonas@gmail.com
"""



from numba import njit,jit
import numpy as np
import numba

GRAVITATIONAL_CONST=6.67430*10**-11

def createMultResBodyfromTopo(bed,datdist=None,thick=None,suplevel=4,multiplier=2,rho_bed=2670.,rho_wath=1030.):
    """
    creates a list of prism-arrays, that change resolution by a given multiplier(usually 2)
    
    Parameters
    ----------

    bed: 2d-array
        array containing height information about the bedrock topography in m
    datdist: 2d-array
        Gives the Kind of data availabil in the grid cell: 0 for ice-free, 1 for batthymetry and 2 for ice-covered
    thick: 2d-array
        ice thickness in m
    suplevel:int
        amount of levels with lower resolution.
        lowest resolution is given with multiplier^suplevel
    multiplier: int
        multiplier for resolution changes, default of 2 should work best
        lowest resolution is given with multiplier^suplevel
    rho_bed: float
        Rock Density in kgm^-3
    rho_wath: float
        Density of sea water
    
    Returns
    -------
    bedprism : List of 3d Arrays
        List contains the 3d arrays/Rock Prism Information of different spatial resolution. 
        The 3d Array contains the upper Boundary [:,:,0] and the lower boundary of the prisms [:,:,1]
    (iceprism) : List of 3d Arrays
        Only returned if thickness is given
        List contains the 3d arrays/Ice Prism Information of different spatial resolution. 
        One 3d Array contains the upper Boundary [:,:,0] and the lower boundary of the prisms [:,:,1]
        If no Ice is present lower and upper Boundary are the same value
    density : List of 2d Arrays
        List contains the 2d arrays/density Information of different spatial resolution. 
        Densitys are later on used to define if the Prisms are corected below sea level(rho_bed-rho_wath) or above sealevel(rho_bed).
    """
    if datdist is None:
        datdist=np.where(bed<0,1,0)
    #cut data before prism creation. Data that cant be assigned to a cell of lowest resolution are deleted
    datdist=np.where(np.logical_and(datdist==1,bed>0),0,datdist)
    supmult=multiplier**suplevel
    bed=bed[:(bed.shape[0]//supmult)*supmult,:(bed.shape[1]//supmult)*supmult]
    
    datdist=datdist[:(bed.shape[0]//supmult)*supmult,:(bed.shape[1]//supmult)*supmult]

    bedprism=[]
    density=[]
    
    bedprism_temp=np.stack((bed,np.zeros(bed.shape)),axis=2)
    density_temp=np.where(datdist==1,-rho_wath+rho_bed,rho_bed)
    density_temp=np.stack((density_temp,np.zeros(density_temp.shape)),axis=2)
    density_temp=density_temp[:,:,0]###??? numba needs magic
    if not thick is None:
        thick=thick[:(bed.shape[0]//supmult)*supmult,:(bed.shape[1]//supmult)*supmult]
        thick=np.where(thick==-9999,0,thick) #can be remove if 0m thicknes is stated with 0
        iceprism=[]
        iceprism_temp=np.stack((bed+thick,bed),axis=2)
        iceprism.append(iceprism_temp)
    bedprism.append(bedprism_temp)
    density.append(density_temp)
    
    for i in np.arange(1,suplevel+1):#creates a prism and density array for every sup level
        j=multiplier**i
        bed2=chunksum(bed,multiplier=j)/j**2
        
        datdist2=datdistRescale(datdist,bed2,j)
        
        

        bedprism_temp=np.stack((bed2,np.zeros(bed2.shape)),axis=2)
        density_temp=np.where(datdist2==1,-rho_wath+rho_bed,rho_bed)
        density_temp=np.stack((density_temp,np.zeros(density_temp.shape)),axis=2)
        density_temp=density_temp[:,:,0]###??? numba needs magic
        if not thick is None:
            thick2=chunksum(thick,multiplier=j)/j**2
            iceprism_temp=np.stack((bed2+thick2,bed2),axis=2)
            iceprism.append(iceprism_temp)
        bedprism.append(bedprism_temp)
        density.append(density_temp)
        
    if not thick is None: 
        return bedprism,iceprism,density
    else: return bedprism,density
    

@njit
def datdistRescale(datdist,bed2,j=2):
    '''
        reduces resolution of the Data distribution Matrix
    Parameters 
    ----------
    datdist : 2d-array
        Matrix containing the datadistribution. 0 for ice-free, 1 for batthymetry and 2 for ice
    bed : 2d-array
        Topography of solid earth .
    j : int
        Gives the factor at which both sides resolution is reduced. The default is 2.

    Returns
    -------
    datdist2 : 2d-Array
        data distribution matrix of reduced resolution.

    '''
    
    datdist2=np.zeros((datdist.shape[0]//j,datdist.shape[1]//j))
    for l in range(datdist.shape[1]//j):
        for k in range(datdist.shape[0]//j):
            #bed2[k,l]=np.mean(bed[k*j:k*j+j,l*j:l*j+j])
            #thick2[k,l]=np.mean(thick[k*j:k*j+j,l*j:l*j+j])
            if np.any(datdist[k*j:k*j+j,l*j:l*j+j]==2):
                datdist2[k,l]=2
            elif np.any(datdist[k*j:k*j+j,l*j:l*j+j]==1) and bed2[k,l]<0:
                datdist2[k,l]=1               
            else:
                datdist2[k,l]=0
    return datdist2

def createrMultResBody(surf1,surf2,suplevel=4,multiplier=2):
    """
        creates a list of prism, that change resolution by a give multiplier(usually 2)
    
    Parameters
    ----------
   
    surf1: 2d-array or number
        array containing height information about the overlaying surface in m
        if surf1<surf2 the gravity calculation will change its sign
    surf2: 2d-array or number
        array containing height information about the unederlaying surface in m
        if surf1<surf2 the gravity calculation will change its sign
    suplevel:int
        amount of levels with lower resolution.
        lowest resolution is given with multiplier^suplevel
    multiplier: int
        multiplier for resolution changes
        lowest resolution is given with multiplier^suplevel
    Returns
    -------
    prisms : List of3d-Array
        Prisms of varying resolution
    """
    supmult=multiplier**suplevel

    if not isinstance(surf1,np.ndarray) and not isinstance(surf1,np.ndarray):
        raise ValueError('at least one surface must be an numpy 2d-array')
    
    if not isinstance(surf1,np.ndarray):surf1=np.ones(surf2.shape)*surf1
    
    if not isinstance(surf2,np.ndarray):surf2=np.ones(surf1.shape)*surf2
    surf1=surf1[:(surf1.shape[0]//supmult)*supmult,:(surf1.shape[1]//supmult)*supmult] #cut data so they fit the biggest cell size
    surf2=surf2[:(surf2.shape[0]//supmult)*supmult,:(surf2.shape[1]//supmult)*supmult]
    prisms=[np.stack((surf1,surf2),axis=2)]
    
    for i in np.arange(1,suplevel+1):#creates a prism and density array for every sup level
        surf1=chunksum(surf1)/4
        surf2=chunksum(surf2)/4
        prisms.append(np.stack((surf1,surf2),axis=2))
    return prisms

def createMultResDensity(density,suplevel=4,multiplier=2):
    """
        creates a list of prism, that change resolution by a give multiplier(usually 2)
    
    Parameters
    ----------
   
    density: 2d-array
        Density values
    suplevel:int
        amount of levels with lower resolution.
        lowest resolution is given with multiplier^suplevel
    multiplier: int
        multiplier for resolution changes
        lowest resolution is given with multiplier^suplevel
    Returns
    -------
    prisms : List of3d-Array
        Prisms of varying resolution
    """
    supmult=multiplier**suplevel

    if not isinstance(density,np.ndarray):
        raise ValueError('density  must be an numpy 2d-array')
    
    
    prisms=[density]
    for i in np.arange(1,suplevel+1):#creates a prism and density array for every sup level
        density=chunksum(density)/4
        prisms.append(density)
    return prisms

def adaptiv_terrain(
    coordinates, prisms,iceprisms, densitys,x,y, field='g_z', dtype="float64", disable_checks=False,
    maxrange=None,threshold=10**-6,createSensMat=False,ref_lev=0,rho_ice=917
):
    """
    Bouger and terrain corection based on 
    Gravitational fields of right-rectangular prisms in Cartesian coordinates
    
    Prisms are used in varying resolution in order to achieve a good runtime as
    a trade-off for lower accuracy. 
    
    The gravitational fields are computed through the analytical solutions
    given by [Nagy2000]_ and [Nagy2002]_, which are valid on the entire domain.
    This means that the computation can be done at any point, either outside or
    inside the prism.

    This implementation makes use of the modified arctangent function proposed
    by [Fukushima2019]_ (eq. 12) so that the potential field to satisfies
    Poisson's equation in the entire domain. Moreover, the logarithm function
    was also modified in order to solve the singularities that the analytical
    solution has on some points (see [Nagy2000]_).

    .. warning::
        The **z direction points upwards**, i.e. positive and negative values
        of ``upward`` represent points above and below the surface,
        respectively. But remember that the ``g_z`` field returns the downward
        component of the gravitational acceleration so that positive density
        contrasts produce positive anomalies.

    Parameters
    ----------
    coordinates : 2d-array
        Array containing ``easting``, ``northing`` and ``upward`` of the
        computation points as arrays, all defined on a Cartesian coordinate
        system and in meters.
    prisms : list of 3d-arrays
        Three dimensional array containing the coordinates of the prisms in the
        following order: west, east, south, north, bottom, top in a Cartesian
        coordinate system. The first two dimensions give the position of a prism.
        The prism informations are aligned along the last dimension. 
        Prismlist is created by createrMultResBody or createMultResBodyfromTopo.
        
        All coordinates should be in meters.
        
    iceprisms : list of 3d-arrays
        Three dimensional array containing the coordinates of the prisms for an ice layer in the
        following order: west, east, south, north, bottom, top in a Cartesian
        coordinate system. The first two dimensions give the position of a prism.
        The prism informations are aligned along the last dimension. 
        Prismlist is created by createrMultResBody or createMultResBodyfromTopo.
        
        All coordinates should be in meters.
        
    density : list of 2d-array
        Array containing the density of each bedrock or ocean prism in kg/m^3 for the differen resolution levels. Must have the
        same size as the number of prisms.
    
    x: 1d-array
        x axis of the Topography data
    y: 1d-array
        y axis of the Topography data
    
    icedensity: int
        density of ice in kg/m^3
    threshold: double
        threshold for stopping the Algorythm to use higher resolution
    createSensMat: bool
        controls wether a sensisitivity matrix is created and returned or not. 
        the snsitivity Matrix will always have the size of the lowest resolution prisms avialabil
    ref_lev: int
        adds a reference layer in a ceratin height given by ref_lev and adds it to the sensitivity matrix 
    
    field : str
        Gravitational field that wants to be computed.
        The available fields are:
        - Downward acceleration: ``g_z``

    dtype : data-type (optional)
        Data type assigned to the resulting gravitational field. Default to
        ``np.float64``.
    disable_checks : bool (optional)
        Flag that controls whether to perform a sanity check on the model.
        Should be set to ``True`` only when it is certain that the input model
        is valid and it does not need to be checked.
        Default to ``False``.

    Returns
    -------
    gravity : 1d-array
        Gravitational field generated by the prisms on the computation points.
    (sensistivity matrix): 2d-array
        sensistivity matrix. Is only returned if createSensMat=true.
    (Output shape): tuple
        resulting shape of the inverted array. Is only returned if createSensMat=true.

    Examples
    --------


    """
    kernels = {"potential": kernel_potential, "g_z": kernel_g_z}
    
    if field not in kernels:
        raise ValueError("Gravitational field {} not recognized".format(field))
    # Figure out the shape and size of the output array
    cast = np.broadcast(*coordinates[:3])
    # Convert coordinates, prisms and density to arrays with proper shape
    #coordinates = tuple(np.atleast_1d(i).ravel() for i in coordinates[:3])
    for prism in prisms:
        prism = np.atleast_3d(prism)
    for density in densitys:
        density = np.atleast_2d(density)
    # Sanity checks
    if not disable_checks:
        for j in range(len(densitys)):
            if densitys[j].shape != (prisms[j].shape[0],prisms[j].shape[1]):
                raise ValueError(
                    "Number of elements in density ({}) ".format(densitys[j].shape)
                    + "mismatch the number of prisms ({})".format(prisms[j].shape)
                )
    # Compute gravitational field
    spacing=x[1]-x[0]
    multiplier=densitys[0].shape[0]//densitys[1].shape[0]
    x=x[:prisms[0].shape[1]]
    y=y[:prisms[0].shape[0]]
    x=np.concatenate((x,np.array(x[-1]+spacing,ndmin=1)))-spacing/2#change x an y value tu the corne vaslues
    y=np.concatenate((y,np.array(y[-1]-spacing,ndmin=1)))+spacing/2
    supx=x[::multiplier**(len(prisms)-1)] # x values of the low-res grid 
    supy=y[::multiplier**(len(prisms)-1)]# y values of the low-res grid 
    

    
    if createSensMat:
        result, sens_mat=adapt_Body_sensmat(coordinates, prisms, densitys,
                                                x,y,threshold=threshold,maxrange=maxrange)
        result +=adapt_Body(coordinates, iceprisms, rho_ice,
                               x,y,threshold=threshold,maxrange=maxrange)
        if ref_lev!=0:
            sens_mat2=fetch_reflev_sens(ref_lev,coordinates,supx,supy,kernels[field])
            sens_mat2=sens_mat2[:sens_mat.shape[0],:sens_mat.shape[1]]
            
            return result.reshape(cast.shape),sens_mat+sens_mat2,(supy.shape[0],supx.shape[0])
        return result.reshape(cast.shape),sens_mat,(supy.shape[0],supx.shape[0])
        
    else:    
        result=adapt_Body(coordinates, prisms, densitys,
                          x,y,threshold=threshold,maxrange=maxrange)
        result +=adapt_Body(coordinates, iceprisms, rho_ice,
                               x,y,threshold=threshold,maxrange=maxrange)
        

    return result.reshape(cast.shape)

def adapt_Body_sensmat(coordinates, prisms, densitys,x,y,threshold=10**-4,field='g_z',sensmatmode='low'):
    '''
    computes the Gravity effect of a body

    Parameters
    ----------
    coordinates : 2d-array
        Array containing ``easting``, ``northing`` and ``upward`` of the
        computation points as arrays, all defined on a Cartesian coordinate
        system and in meters.
    prisms : list of 3d-arrays
        Three dimensional array containing the coordinates of the prisms in the
        following order: bottom, top in a Cartesian
        coordinate system. 
        The prism informations are aligned along the last dimension. 
        Prismlist is created by createrMultResBody or createMultResBodyfromTopo.
        
        All coordinates should be in meters.
    densitys : list of 2d-array or float
        Array containing the density of each bedrock or ocean prism in kg/m^3. Must have the
        same size as the number of prisms.
    x: 1d-array
        x axis of the Topography data.
    y: 1d-array
        y axis of the Topography data.
    threshold : float, optional
        Gravity threshold for the break of a topography assignment in mgal. The default is 10**-6 mgal.

    Returns
    -------
    1d-array
        gravity result of the computation.
    2d-array
        sensistivity matrix.

    '''
    kernels = {"potential": kernel_potential, "g_z": kernel_g_z, "top_gradient": kernel_gz_surfgradient}
    prismfunc = {"potential": jit_prism, "g_z": jit_prism, "top_gradient": jit_prism_top}
    if field not in kernels:
        raise ValueError("Gravitational field {} not recognized".format(field))
    
    
    if prisms[0].shape[1]!=x.shape[0]-1 or prisms[0].shape[0]!=y.shape[0]-1: #adjust x and y acoording to given format(cell/corner)
        spacing=x[1]-x[0]
        x=x[:prisms[0].shape[1]]
        y=y[:prisms[0].shape[0]]
        x=np.concatenate((x,np.array(x[-1]+spacing,ndmin=1)))-spacing/2#change x an y value tu the corne vaslues
        y=np.concatenate((y,np.array(y[-1]-spacing,ndmin=1)))-spacing/2
     
    
    prisms=tuple(prisms)
    try:#if only one density is given, it will use it later on as the only density
        densitys[0]
        densitys=tuple(densitys)
        selector=selector_var
    except:
        selector=selector_cons
    out,sensmat=jit_adapt_Body_sensmat(coordinates, prisms, densitys,x,y,threshold,selector,kernels[field],prismfunc[field],sensmatmode=sensmatmode)
    return out*1e5*GRAVITATIONAL_CONST,sensmat*1e5*GRAVITATIONAL_CONST
    

    

@jit(nopython=True)
def jit_adapt_Body_sensmat(
    coordinates, prisms, densitys,x,y,
    error,selector,kernel,prismfunc,sensmatmode='low'):  # pylint: disable=invalid-name
    """
    Compute gravitational field of a prismset and returns the sensitivity matrix for the lowest resolution matrix availabil
    the resulution varies depending on the impact of the prism.

    Parameters
    ----------
    coordinates : 2d-array
        Array containing ``easting``, ``northing`` and ``upward`` of the
        computation points as arrays, all defined on a Cartesian coordinate
        system and in meters.
    prisms : list of 3d-arrays
         Three dimensional array containing the coordinates of the prisms in the
         following order: bottom, top in a Cartesian
         coordinate system. 
         The prism informations are aligned along the last dimension. 
         Prismlist is created by createrMultResBody or createMultResBodyfromTopo.
        
        All coordinates should be in meters.
        
    iceprisms : list of 3d-arrays
        Three dimensional array containing the coordinates of the prisms in the
        following order: bottom, top in a Cartesian
        coordinate system. 
        The prism informations are aligned along the last dimension. 
        Prismlist is created by createrMultResBody or createMultResBodyfromTopo.
        
        All coordinates should be in meters.
        
    densitys : list of 2d-array or float
        Array containing the density of each bedrock or ocean prism in kg/m^3. Must have the
        same size as the number of prisms.
        
        
    supx: 1d-Array
        x-coordinate vector fo the lowest resolution grid contained in prisms
    supy: 1d-Array
        y-coordinate vector fo the lowest resolution grid contained in prisms
    icedensity: int
        density of ice in kg/m^3
    error: double
        threshold in mgal for stopping the Algorythm to use higher resolution
    kernel : func
        Kernel function that will be used to compute the desired field.
    
    Returns
    -------
    out : 1d-array
        Array where the resulting gravity values will be stored.
    sens_mat: 2d-array
        Sensistivity Matrix
    """
    out=np.zeros(coordinates.shape[1])
    levels=len(prisms)
    multiplier=prisms[0].shape[0]//prisms[1].shape[0]

    
    supx=prisms[-1].shape[1]#len(x[::multiplier**(len(prisms)-1)])#-1 # amount values in the low-res grid x dimension
    supy=prisms[-1].shape[0]#len(y[::multiplier**(len(prisms)-1)])#-1# amount values in the low-res grid y dimension
    
    stats=np.zeros(len(prisms))
    error=error/(GRAVITATIONAL_CONST*1e5)#*relrad**2)#calculate allowed error per cell in low-res
    grav_buf=[]
    valid=[]
    #create list of Arrays for later use in gravity calculations
    # they don have any usage except to store calculated data conviniently
    for j in range(levels):
        grav_buf.append(np.zeros((multiplier**j,multiplier**j)))
        valid.append(np.zeros((multiplier**j,multiplier**j),dtype=np.dtype('?')))
    sens_buf=grav_buf.copy()
    
    if sensmatmode=='low':
        #sens_mat=np.zeros((coordinates[0].size,supx*supy))
        sens_mat=np.zeros((supy,supx,coordinates[0].size))
    else:
        sens_mat=np.ones((prisms[0].shape[0],prisms[0].shape[1],coordinates[0].size))
        
    #print(sens_mat.shape)
    print((prisms[0].shape[0],prisms[0].shape[1]))
    for l in range(coordinates[0].size):

        #iterates over low-res grid
        
        for xind_it in range(supx):#range(xind-relrad,xind+relrad):
            for yind_it in range(supy):#range(yind-relrad,yind+relrad):
                # if xind_it>supy-1:
                #     print('stop')
                grav=0
                valid[0]=np.array([[True]])
                for m in range(len(prisms)):
                    gridspac=multiplier**(levels-1-m)
                    #selcting the prisms matching to superior cell level and filtered values  
                    cur_prisms=prisms[-m-1][yind_it*multiplier**m:yind_it*multiplier**m+multiplier**m,
                                            xind_it*multiplier**m:xind_it*multiplier**m+multiplier**m,:]
                    
                    valid[m]=np.logical_and((cur_prisms[:,:,0]-cur_prisms[:,:,1])!=0,valid[m])
                    
                    cur_x=x[::gridspac][xind_it*multiplier**m:(xind_it+1)*multiplier**m+multiplier**m]
                    cur_y=y[::gridspac][yind_it*multiplier**m:(yind_it+1)*multiplier**m+multiplier**m]
                    
                    for j in range(cur_prisms.shape[0]):
                        for k in range(cur_prisms.shape[1]):
                            if valid[m][j,k]:
                                
                                prism=np.array([cur_x[k],cur_x[k+1],
                                                cur_y[j],cur_y[j+1],cur_prisms[j,k,1],cur_prisms[j,k,0]])
                                density=selector(densitys,yind_it,xind_it,multiplier,j,k,m)
                                grav_buf[m][j,k]=prismfunc(prism,density,coordinates[:,l],kernel)
                                sens_buf[m][j,k]=grav_buf[m][j,k]
                                
                    #compare grav value to previous value and decide wich areas can be improved
                    if np.any(valid[m])==False: #breaks the loop if all values are already calculated
                            break
                    if m>0 and m<levels-1:
                        pos=np.argwhere(valid[m-1])

                        for k in range(pos.shape[0]):
                            
                            gravsum=np.sum(grav_buf[m][pos[k,0]*multiplier:pos[k,0]*multiplier+multiplier,
                                                       pos[k,1]*multiplier:pos[k,1]*multiplier+multiplier])
                            t=error#/(multiplier**(2*(m-1)))
                            if abs(grav_buf[m-1][(pos[k,0],pos[k,1])]-gravsum)<t:
                                grav+=gravsum
                                stats[m]+=1/(supx*supy*4**m)
                                
                                #set values that are precise enough to False to ignore them in further loops
                                valid[m][pos[k,0]*multiplier:pos[k,0]*multiplier+multiplier,
                                         pos[k,1]*multiplier:pos[k,1]*multiplier+multiplier]=False 
                        valid[m+1]=enlargeBoolArray(valid[m])
                        sens_buf[m+1]=enlargeArray_arithmetic(sens_buf[m])
                    elif(m==0):
                        t=error
                        if(abs(grav_buf[m][0,0])<t): 
                            stats[m]+=1/(supx*supy*4**m)
                            grav+=grav_buf[m][0,0]
                            valid[m][0,0]=False
                            
                            break
                        valid[m+1]=enlargeBoolArray(valid[m])
                        sens_buf[m+1]=enlargeArray_arithmetic(sens_buf[m])
                    elif m==levels-1:#on the last level all results that are still valid have to be added to the main grav-value
                        pos=np.argwhere(valid[m])
                        for k in range(pos.shape[0]):
                            grav+=grav_buf[m][(pos[k,0],pos[k,1])]
                            stats[m]+=1/(supx*supy*4**m)
                out[l]+=grav
                
                if sensmatmode=='low':
                    #sens_mat[l,xind_it+yind_it*supx]=np.sum(sens_buf[m])
                    sens_mat[yind_it,xind_it,l]=np.sum(sens_buf[m])
                else:
                    if m!=levels-1:
                        sens_buf[-1]=enlargeArray_arithmetic(sens_buf[m],multiplier=2**(levels-1-m))
                    xpos=xind_it*multiplier**(levels-1)
                    ypos=yind_it*multiplier**(levels-1)
                    
                    sens_mat[ypos:ypos+multiplier**(levels-1),xpos:xpos+multiplier**(levels-1),l]=sens_buf[-1]#.T
                    # for k in range(sens_buf[-1].shape[1]):
                    #     senspos=(xind_it*multiplier**(levels-1)+
                    #              yind_it*prisms[0].shape[0]*multiplier**(levels-1)+k)
                    #     sens_mat[l,senspos:senspos+multiplier**(levels-1)]=sens_buf[-1][k]
    #stats=np.array(stats)/np.array([1,4,16,64,4096])
    
    print(stats)
    return out,sens_mat.reshape(sens_mat.shape[0]*sens_mat.shape[1],sens_mat.shape[2]).T


def sensmat_from_mesh(coordinates, prisms, density,includeDens=False,relev=-10_000):
    if isinstance(density,np.ndarray):
        if density.shape[0]>1:
            selector=selector_var_mesh#lambda d: density[d]
        else:
            selector=selector_cons_mesh#lambda d: density[0]
            density=density[0]
    else:
        selector=selector_cons_mesh#lambda d: density
    sensmat=sensmat_from_mesh_jit(coordinates, prisms, density,selector)
    #print('test')
    if includeDens:
        sensmat_d=sensmat_dens_from_mesh_jit(coordinates, prisms, relev)
        #raise ValueError
        sensmat=np.column_stack((sensmat,sensmat_d))
        #sensmat=np.append(sensmat,sensmat_d,axis=0)
    return sensmat
@njit
def sensmat_from_mesh_jit(coordinates, prisms, density,selector):

    if prisms.shape[0]==6:
        sens_mat=np.zeros((coordinates[0].size,prisms.shape[0]))
        for l in range(coordinates[0].size):
    
        #iterates over grid
            for k in range(prisms.shape[0]):#range(xind-relrad,xind+relrad):
                den=selector(density,k)
                #print(den,selector)
                
                grav=jit_prism_top(list(prisms[k,:]),den,coordinates[:,l],kernel_gz_surfgradient)
                sens_mat[l,k]=grav
    else:
        sens_mat=np.zeros((coordinates[0].size,prisms.shape[0]))
        for l in range(coordinates[0].size):
    
        #iterates over grid
            for k in range(prisms.shape[0]):#range(xind-relrad,xind+relrad):
                den=selector(density,k)
                #print(den,selector)
                grav=jit_prism_top(list(prisms[k,:4])+[0,prisms[k,4]],den,coordinates[:,l],kernel_gz_surfgradient)
                sens_mat[l,k]=grav
    return sens_mat*1e5*GRAVITATIONAL_CONST

@njit
def sensmat_dens_from_mesh_jit(coordinates, prisms,reflev=None):

    if prisms.shape[1]==6:
        sens_mat=np.zeros((coordinates[0].size,prisms.shape[0]))
        for l in range(coordinates[0].size):
    
        #iterates over grid
            for k in range(prisms.shape[0]):
                #den=selector(density,k)
                
                grav=jit_prism(list(prisms[k,:]),1,coordinates[:,l],kernel_g_z)
                sens_mat[l,k]=grav
    else:
        sens_mat=np.zeros((coordinates[0].size,prisms.shape[0]))
        for l in range(coordinates[0].size):
    
        #iterates over grid
            for k in range(prisms.shape[0]):
                #den=selector(density,k)
                grav=jit_prism(list(prisms[k,:4])+[reflev,prisms[k,4]],1,coordinates[:,l],kernel_g_z)
                sens_mat[l,k]=grav
    return sens_mat*1e5*GRAVITATIONAL_CONST

@njit
def selector_cons_mesh(density,d):
    return density
@njit
def selector_var_mesh(density,d):
    return density[d]


def grav_from_surfmesh(coordinates, prisms, density,reflev=0):
    if isinstance(density,np.ndarray):
        if density.shape[0]>1:
            selector=selector_var_mesh#lambda d: density[d]
        else:
            selector=selector_cons_mesh#lambda d: density[0]
            density=density[0]
    else:
        selector=selector_cons_mesh#lambda d: density
    return grav_from_surfmesh_jit(coordinates, prisms, density,selector,reflev=reflev)

@njit
def grav_from_surfmesh_jit(coordinates, prisms, density,selector,reflev):
    
    grav=np.zeros(coordinates.shape[1])
    p=np.zeros(6)-reflev
    if prisms.shape[1]==6:
        for l in range(coordinates[0].size):
        #iterates over grid
            for k in range(prisms.shape[0]):#range(xind-relrad,xind+relrad):
                
                grav[l]+=jit_prism(prisms[k,:],selector(density,k),coordinates[:,l],kernel_g_z)
    else:
        for l in range(coordinates[0].size):
        #iterates over grid
            for k in range(prisms.shape[0]):#range(xind-relrad,xind+relrad):
                p[:4]=prisms[k,:4]
                p[5]=prisms[k,4]
                
                grav[l]+=jit_prism(p,selector(density,k),coordinates[:,l],kernel_g_z)
            #sens_mat[l,k]=grav[l]
    return grav*1e5*GRAVITATIONAL_CONST


def grav_from_mesh(coordinates, prisms, density):
    if isinstance(density,np.ndarray):
        if density.shape[0]>1:
            selector=selector_var_mesh#lambda d: density[d]
        else:
            selector=selector_cons_mesh#lambda d: density[0]
            density=density[0]
    else:
        selector=selector_cons_mesh#lambda d: density
    return grav_from_mesh_jit(coordinates, prisms, density,selector)

@njit
def grav_from_mesh_jit(coordinates, prisms, density,selector):
    
    grav=np.zeros(coordinates.shape[1])
    if prisms.shape[1]==6:
        for l in range(coordinates[0].size):
        #iterates over grid
            for k in range(prisms.shape[0]):#range(xind-relrad,xind+relrad):
                
                grav[l]+=jit_prism(prisms[k,:],selector(density,k),coordinates[:,l],kernel_g_z)
        
    return grav*1e5*GRAVITATIONAL_CONST

@njit
def create_sensmat_interface(coordinates, prisms, density):
    supx=prisms[0,:,0]
    supy=prisms[:,0,3]
    sens_mat=np.zeros((coordinates[0].size,len(supx)*len(supy)))
    for l in range(coordinates[0].size):

    #iterates over grid
        for xind in range(len(supx)):#range(xind-relrad,xind+relrad):
            for yind in range(len(supy)):#range(yind-relrad,yind+relrad):
                grav=jit_prism_top(prisms[yind,xind,:],density,coordinates[:,l],kernel_gz_surfgradient)
                sens_mat[l,xind+yind*len(supx)]=grav
    return sens_mat*1e5*GRAVITATIONAL_CONST

#Just two selctor functions for variable and constant density, should save a bit of space
@njit
def selector_cons(densitys,yind_it,xind_it,multiplier,j,k,m):
    return densitys 

@njit
def selector_var(densitys,yind_it,xind_it,multiplier,j,k,m):
    density=densitys[-m-1][yind_it*multiplier**m:yind_it*multiplier**m+multiplier**m,
                           xind_it*multiplier**m:xind_it*multiplier**m+multiplier**m][j,k]
    return density

@jit(nopython=True)
def fetch_reflev_sens(reflev,coordinates,supx,supy,kernel):
    '''
    Gives the values which needs to be added to a sensistivity Matrix to refer to a lower boundary

    Parameters
    ----------
    reflev : float
        adjusted reference Level.
    coordinates : 2d-array
        Array containing ``easting``, ``northing`` and ``upward`` of the
        computation points as arrays, all defined on a Cartesian coordinate
        system and in meters.
    supx : 1d-array
        x axis of the sensistivity matrix
    supy : 1d-array
        x axis of the sensistivity matrix
    kernel : func
        DESCRIPTION.

    Returns
    -------
    2-d Matrix:
        Sensistivity Matrix.

    '''
    cs=supx[1]-supx[0]
    sens_mat=np.zeros((coordinates[0].size,len(supx)*len(supy)))
    for l in range(coordinates[0].size):
        
        for xind in range(len(supx)):#range(xind-relrad,xind+relrad):
            for yind in range(len(supy)):#range(yind-relrad,yind+relrad):
                sens_mat[l,xind+yind*len(supx)]=jit_prism(
                    np.array([supx[xind]-cs/2, supx[xind]+cs/2, supy[yind]-cs/2, supy[yind]+cs/2, reflev, 0]),
                    1,coordinates[:,l],kernel)
    return sens_mat*1e5*GRAVITATIONAL_CONST
        
        
def adapt_Body(coordinates, prisms, densitys,x,y,threshold=10**-4,maxrange=None):
    '''
    computes the Gravity effect of a body

    Parameters
    ----------
    coordinates : 2d-array
        Array containing ``easting``, ``northing`` and ``upward`` of the
        computation points as arrays, all defined on a Cartesian coordinate
        system and in meters.
    prisms : list of 3d-arrays
         Three dimensional array containing the coordinates of the prisms in the
         following order: bottom, top in a Cartesian
         coordinate system. 
         The prism informations are aligned along the last dimension. 
         Prismlist is created by createrMultResBody or createMultResBodyfromTopo.
        
        All coordinates should be in meters.
    densitys : list of 2d-array or float
        Array containing the density of each bedrock or ocean prism in kg/m^3. Must have the
        same size as the number of prisms.
    x: 1d-array
        x axis of the Topography data (highest resolution level).
    y: 1d-array
        y axis of the Topography data (highest resolution level).
    threshold : float, optional
        Gravity threshold for the break of a topography assignment in mgal. The default is 10**-6 mgal.

    Returns
    -------
    1d-array
        gravity result of the computation.

    '''
    
    if prisms[0].shape[1]!=x.shape[0]-1 or prisms[0].shape[0]!=y.shape[0]-1:
        spacing=np.abs(x[1]-x[0])
        x=x[:prisms[0].shape[1]]
        y=y[:prisms[0].shape[0]]
        x=np.concatenate((x,np.array(x[-1]+spacing,ndmin=1)))-spacing/2#change x an y value tu the corne vaslues
        y=np.concatenate((y,np.array(y[-1]-spacing,ndmin=1)))-spacing/2
    
    prisms=tuple(prisms)
    try:#if only one density is given, it will use it later on as the only density
        densitys[0]
        densitys=tuple(densitys)
        selector=selector_var
    except:
        selector=selector_cons
    out,stats=jit_adapt_Body(coordinates, prisms, densitys,x,y,threshold,selector,maxrange=maxrange)
    stats=stats*100
    labeled_stats=['level 0 (lowest resolution):  '+str(stats[0])]
    for k in range (1,len(stats)-1): labeled_stats.append('level ' + str(k)+':  '+str(stats[k]))
    labeled_stats.append('level ' + str(k+1)+ ' (highest resolution):  ' +str(stats[-1]))
    for k in range(len(labeled_stats)):print(labeled_stats[k])
    print('(percentage of finished area)')
    
    return out*1e5*GRAVITATIONAL_CONST
    
    
@jit(nopython=True)
def jit_adapt_Body(
    coordinates, prisms, densitys,x,y,
    error,selector,maxrange=None):  # pylint: disable=invalid-name
    """
    Compute gravitational field of a prismset and returns the sensitivity matrix for the lowest resolution matrix availabil
    the resulution varies depending on the impact of the prism.

    Parameters
    ----------
    coordinates : Tuple
        Tuple containing ``easting``, ``northing`` and ``upward`` of the
        computation points as arrays, all defined on a Cartesian coordinate
        system and in meters.
    prisms : list of 3d-arrays
        Three dimensional array containing the coordinates of the prisms in the
        following order: bottom, top in a Cartesian
        coordinate system. 
        The prism informations are aligned along the last dimension. 
        Prismlist is created by createrMultResBody or createMultResBodyfromTopo.
        
        All coordinates should be in meters.
        
    iceprisms : list of 3d-arrays
        Three dimensional array containing the coordinates of the prisms for an ice layer in the
        following order: west, east, south, north, bottom, top in a Cartesian
        coordinate system. The first two dimensions give the position of a prism.
        The prism informations are aligned along the last dimension. 
        Prismlist is created by createSupPrism.
        
        All coordinates should be in meters.
        
    density : 2d-array
        Array containing the density of each bedrock or ocean prism in kg/m^3. Must have the
        same size as the number of prisms.

        
    x: 1d-Array
        x-coordinate vector for the highest resolution grid contained in prisms
    y: 1d-Array
        y-coordinate vector for the highest resolution grid contained in prisms
    icedensity: int
        density of ice in kg/m^3
    error: double
        threshold in mgal for stopping the Algorythm to use higher resolution
    kernel : func
        Kernel function that will be used to compute the desired field.
    out : 2d-array
        Array where the resulting field values will be stored.
        Sensitivitymatrix
    """
    out=np.zeros(coordinates.shape[1])
    levels=len(prisms)
    multiplier=prisms[0].shape[0]//prisms[1].shape[0]

    
    supx=len(x[::multiplier**(len(prisms)-1)])-1 # amount values in the low-res grid x dimension
    supy=len(y[::multiplier**(len(prisms)-1)])-1# amount values in the low-res grid y dimension
    
    stats=np.zeros(len(prisms),dtype=np.double)
    error=error/(GRAVITATIONAL_CONST*1e5)#*relrad**2)#calculate allowed error per cell in low-res
    grav_buf=[]
    valid=[]
    spacing=abs(x[1]-x[0])
    if not maxrange is None: relrad=int(np.ceil(maxrange/(spacing*multiplier**(levels-1))))
    
    #create list of Arrays for later use in gravity calculations
    # they don have any usage except to store calculatet data conviniently
    for j in range(levels):
        grav_buf.append(np.zeros((multiplier**j,multiplier**j)))
        valid.append(np.zeros((multiplier**j,multiplier**j),dtype=np.dtype('?')))
    
    for l in range(coordinates[0].size):
        
        if maxrange is None:
            xl=0
            xu=supx
            yl=0
            yu=supy
        else:   
            xind=np.argwhere(np.abs(coordinates[0,l]-x[::multiplier**(levels-1)])
                             <=spacing*multiplier**(levels-1)/2)[0,0]
            yind=np.argwhere(np.abs(coordinates[1,l]-y[::multiplier**(levels-1)])
                             <=spacing*multiplier**(levels-1)/2)[0,0]
            xl=0 if xind-relrad<0 else xind-relrad
            xu=supx if xind+relrad>supx else xind+relrad
            yl=0 if yind-relrad<0 else yind-relrad
            yu=supy if yind+relrad>supy else yind+relrad
        #iterates over low-res grid
        for xind_it in range(xl,xu):#range(supx):
            for yind_it in range(yl,yu):#range(supy):
                size_workarea = (xu-xl) *(yu-yl)
                #if size_workarea!=450:print((xu,xl,xind,relrad),(yu,yl),(xu-xl-1) *(yu-yl-1))
                grav=0
                
                valid[0]=np.array([[True]])
                for m in range(len(prisms)):
                    
                    gridspac=multiplier**(levels-1-m)
                    #selcting the prisms matching to superior cell level and filtered values  
                    cur_prisms=prisms[-m-1][yind_it*multiplier**m:yind_it*multiplier**m+multiplier**m,
                                            xind_it*multiplier**m:xind_it*multiplier**m+multiplier**m,:]
                    tmp=np.logical_and((cur_prisms[:,:,0]-cur_prisms[:,:,1])!=0,valid[m])
                    
                        
                    stats[m]+=np.double(np.sum(valid[m]!=tmp))/np.double((size_workarea*coordinates.shape[1])
                                              )*(np.double(4)**np.double(-m))*(np.double(4)**np.double(-m))
                    valid[m]=tmp
                    
                    cur_x=x[::gridspac][xind_it*multiplier**m:(xind_it+1)*multiplier**m+multiplier**m]
                    cur_y=y[::gridspac][yind_it*multiplier**m:(yind_it+1)*multiplier**m+multiplier**m]
                    
                    for j in range(cur_prisms.shape[0]):
                        for k in range(cur_prisms.shape[1]):
                            if valid[m][j,k]:
                                
                                prism=np.array([cur_x[k],cur_x[k+1],
                                                cur_y[j],cur_y[j+1],cur_prisms[j,k,1],cur_prisms[j,k,0]])
                                density=selector(densitys,yind_it,xind_it,multiplier,j,k,m)
                                
                                
                                # if len(prisms)-1==m and l==0: test[yind_it*multiplier**m:yind_it*multiplier**m+multiplier**m,
                                #             xind_it*multiplier**m:xind_it*multiplier**m+multiplier**m][j,k]=jit_prism(prism,density,coordinates[:,l],kernel_g_z)
                                grav_buf[m][j,k]=jit_prism(prism,density,coordinates[:,l],kernel_g_z)
                                
                                
                    #compare grav value to previous value and decide wich areas can be improved
                    if np.any(valid[m])==False: #breaks the loop if all values are already calculated
                            break
                    if m>0 and m<levels-1:
                        pos=np.argwhere(valid[m-1])

                        for k in range(pos.shape[0]):
                            
                            gravsum=np.sum(grav_buf[m][pos[k,0]*multiplier:pos[k,0]*multiplier+multiplier,
                                                       pos[k,1]*multiplier:pos[k,1]*multiplier+multiplier])
                            t=error#/(multiplier**(2*(m-1)))
                            
                            if abs(grav_buf[m-1][(pos[k,0],pos[k,1])]-gravsum)<t:
                                stats[m]+=1/np.double((size_workarea*coordinates.shape[1])
                                                          )*(np.double(4)**np.double(-m+1))
                                grav+=gravsum
                                #set values that are precise enough to False to ignore them in further loops
                                valid[m][pos[k,0]*multiplier:pos[k,0]*multiplier+multiplier,
                                         pos[k,1]*multiplier:pos[k,1]*multiplier+multiplier]=False 
                        valid[m+1]=enlargeBoolArray(valid[m])
                        
                    elif(m==0):
                        
                        t=error
                        if(abs(grav_buf[m][0,0])<t): 
                            stats[m]+=1/np.double((size_workarea*coordinates.shape[1])
                                                      )*(np.double(4)**-np.double(m))
                            grav+=grav_buf[m][0,0]
                            valid[m][0,0]=False
                            
                            break
                        valid[m+1]=enlargeBoolArray(valid[m])
                    elif m==levels-1:#on the last level all results that are still valid have to be added to the main grav-value
                        pos=np.argwhere(valid[m])
                        for k in range(pos.shape[0]):
                            grav+=grav_buf[m][(pos[k,0],pos[k,1])]
                            stats[m]+=1/np.double((size_workarea*coordinates.shape[1])
                                                      )*(np.double(4)**np.double(-m))
                out[l]+=grav

    return out,stats

@njit
def create_sensmat(coordinates, prisms, density):
    supx=prisms[0,:,0]
    supy=prisms[:,0,3]
    sens_mat=np.zeros((coordinates[0].size,len(supx)*len(supy)))
    for l in range(coordinates[0].size):

    #iterates over grid
        for xind in range(len(supx)):#range(xind-relrad,xind+relrad):
            for yind in range(len(supy)):#range(yind-relrad,yind+relrad):
                grav=jit_prism(prisms[yind,xind,:],density,coordinates[:,l],kernel_g_z)
                sens_mat[l,xind+yind*len(supx)]=grav
    return sens_mat*1e5*GRAVITATIONAL_CONST



@njit
def jit_prism(prism,density,coordinate,kernel):
    '''
    -->Code by Harmonica
    '''
    
    out=0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                shift_east = prism[1 - i]
                shift_north = prism[ 3 - j]
                shift_upward = prism[ 5 - k]
                # If i, j or k is 1, the shift_* will refer to the
                # lower boundary, meaning the corresponding term should
                # have a minus sign
                out += (
                    density
                    * (-1) ** (i + j + k)
                    * kernel(
                        shift_east - coordinate[0],
                        shift_north - coordinate[1],
                        shift_upward - coordinate[2],
                    )
                )
    return out

@njit
def jit_prism_top(prism,density,coordinate,kernel):
    '''
    
    '''
    k=0
    out=0
    for i in range(2):
        for j in range(2):
            
            shift_east = prism[1 - i]
            shift_north = prism[ 3 - j]
            shift_upward = prism[ 5-k]
            # If i, j or k is 1, the shift_* will refer to the
            # lower boundary, meaning the corresponding term should
            # have a minus sign
            out += (
                density
                * (-1) ** (i + j + k)
                * kernel(
                    shift_east - coordinate[0],
                    shift_north - coordinate[1],
                    shift_upward - coordinate[2],
                )
            )
    return out

@njit
def create_sensmat_interface(coordinates, prisms, density):
    supx=prisms[0,:,0]
    supy=prisms[:,0,3]
    sens_mat=np.zeros((coordinates[0].size,len(supx)*len(supy)))
    for l in range(coordinates[0].size):

    #iterates over grid
        for xind in range(len(supx)):#range(xind-relrad,xind+relrad):
            for yind in range(len(supy)):#range(yind-relrad,yind+relrad):
                grav=jit_prism_top(prisms[yind,xind,:],density,coordinates[:,l],kernel_gz_surfgradient)
                sens_mat[l,xind+yind*len(supx)]=grav
    return sens_mat*1e5*GRAVITATIONAL_CONST


# @jit(nopython=True,cache=False)
# def kernel_gz_surfgradient(easting, northing, upward):
#     """
#     Kernel for downward component of gravitational acceleration of a prism
#     """
#     radius = np.sqrt(easting ** 2 + northing ** 2 + upward ** 2)
#     kernel = (
#         (northing*upward)/(radius*(radius+easting))
#         +(upward*easting)/(radius*(radius+northing))
#         -upward*(-easting*northing/(upward**2*radius)-easting*northing*radius**-3)
#         /(easting**2*northing**2/(upward**2*radius**2)+1)
#         - safe_atan2(easting * northing, upward * radius)
#     )
#     return kernel

@jit(nopython=True,cache=False)
def kernel_gz_surfgradient(x, y, z):
    """
    Kernel for downward component of gravitational acceleration of a prism
    """
    radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    kernel = (
        z*(x/(x**2+z**2)+y/(y**2+z**2))
        - safe_atan2(x * y, z * radius)
    )
    return kernel

@jit(nopython=True,cache=False)
def kernel_potential(easting, northing, upward):
    """
    Kernel function for potential gravitational field generated by a prism
    -->Code by Harmonica
    """
    radius = np.sqrt(easting ** 2 + northing ** 2 + upward ** 2)
    kernel = (
        easting * northing * safe_log(upward + radius)
        + northing * upward * safe_log(easting + radius)
        + easting * upward * safe_log(northing + radius)
        - 0.5 * easting ** 2 * safe_atan2(upward * northing, easting * radius)
        - 0.5 * northing ** 2 * safe_atan2(upward * easting, northing * radius)
        - 0.5 * upward ** 2 * safe_atan2(easting * northing, upward * radius)
    )
    return kernel


@jit(nopython=True,cache=False)
def kernel_g_z(easting, northing, upward):
    """
    Kernel for downward component of gravitational acceleration of a prism
    
    -->Code by Harmonica
    """
    radius = np.sqrt(easting ** 2 + northing ** 2 + upward ** 2)
    kernel = (
        easting * safe_log(northing + radius)
        + northing * safe_log(easting + radius)
        - upward * safe_atan2(easting * northing, upward * radius)
    )
    return kernel


@jit(nopython=True,cache=False)
def safe_atan2(y, x):
    """
    Principal value of the arctangent expressed as a two variable function

    This modification has to be made to the arctangent function so the
    gravitational field of the prism satisfies the Poisson's equation.
    Therefore, it guarantees that the fields satisfies the symmetry properties
    of the prism. This modified function has been defined according to
    [Fukushima2019]_.
    
    -->Code by Harmonica
    """
    if x != 0:
        result = np.arctan(y / x)
    else:
        if y > 0:
            result = np.pi / 2
        elif y < 0:
            result = -np.pi / 2
        else:
            result = 0
    return result


@jit(nopython=True,cache=False)
def safe_log(x):
    """
    Modified log to return 0 for log(0).
    The limits in the formula terms tend to 0 (see [Nagy2000]_).
    
    -->Code by Harmonica
    """
    if np.abs(x) < 1e-10:
        result = 0
    else:
        result = np.log(x)
    return result
@njit
def chunksum(array,multiplier=2):
    '''
    
    Sums multiplier x multiplier chunks of an array
    

    Parameters
    ----------
    array : 2d-Array
        input Array.
    multiplier : int, optional
        Size of one chunk. The default is 2.

    Returns
    -------
    Sum : 2d-Array
        chunk summed array, size is reduced by the multiplier.

    '''
    
    
    preSum=array[::multiplier,:].copy()
    for j in range(1,multiplier):
        preSum+=array[j::multiplier,:]
    Sum=preSum[:,::multiplier]
    for j in range(1,multiplier):
        Sum+=preSum[:,j::multiplier]
    return Sum

@njit
def enlargeBoolArray(array,multiplier=2):
    '''
    Enlarges a boolean array by a given int Factor

    Parameters
    ----------
    array : 2d-array (boolean)
        input array.
    multiplier : int, optional
        The Factor by which the Array is enlarged. The default is 2.

    Returns
    -------
    result : 2d-array (boolean)
        Enlarged Boolean Array.

    '''
    
    resshape=(array.shape[0]*multiplier,array.shape[1]*multiplier)#shape of the result array
    result=np.zeros(resshape,dtype=np.dtype('?'))
    for j in range(multiplier):
        for k in range(multiplier):
            result[j::multiplier,k::multiplier]=array
    return result

@njit
def enlargeArray_arithmetic(array,multiplier=2):
    '''
    enlarges an array by a given, but keeping the sum constant

    Parameters
    ----------
    array : 2d-array
        input Array.
    multiplier : int, optional
        The Factor by which the Array is enlarged. The default is 2.

    Returns
    -------
    result : 2d-array
        enlarged Array.

    '''
    resshape=(array.shape[0]*multiplier,array.shape[1]*multiplier)#shape of the result array
    result=np.zeros(resshape)
    for j in range(multiplier):
        for k in range(multiplier):
            result[j::multiplier,k::multiplier]=array/(multiplier**2)#to keep the sum equal to the overall gravity effect, a division is necesarry
    return result





'''A simple example:'''
#%%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def randomVar(shape,sigma=10):
        """creates a random normal distributed Topography"""
        from scipy.ndimage import gaussian_filter
        val=np.random.randn(shape[0]*shape[1])
        val=val.reshape(shape)
        grid=gaussian_filter(val,sigma=sigma)
        grid=grid*(1/np.std(grid))
        return grid
    
    """Simple Topgraphic Gravity Effect"""
    x=np.arange(400)*100 #xaxis of the data
    y=np.arange(300)*100 #yaxis of the data
    X,Y=np.meshgrid(x,y)
    topo=randomVar((x.shape[0],y.shape[0]),sigma=20).T*500# topography data

    xc=x[50:-50:5]
    yc=y[50:-50:5]
    coordx,coordy=np.meshgrid(x[50:-50:5],y[50:-50:5])
    coords=np.array([coordx.flatten(),coordy.flatten(),3000*np.ones(coordy.flatten().shape[0])])#List of coordinates (easting, northing, height)
    #coords=np.array([coordx.flatten(),coordy.flatten(),topo[50:-50:5,50:-50:5].flatten()+1000])
    
    topoprisms,densities=createMultResBodyfromTopo(topo,suplevel=4) #prepare the Topography
    #Topography is now used in multiple resolutions
    
    grav=adapt_Body(coords,topoprisms,densities,x,y,threshold=10**-4)
    
    _,(pl1,pl2)=plt.subplots(2)
    
    plot1=pl1.imshow(topo, extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
    cbar=plt.colorbar(plot1,ax=pl1)
    pl1.set_ylabel('Topoography height in m')
    
    pl1.scatter(coords[0,:],coords[1,:],s=3,marker='+',color=[1,1,1])
    
    
    #plt.imshow(topo, extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
    plot1=pl2.imshow(grav.reshape((coordx.shape[0],coordx.shape[1])),extent=[np.min(xc),np.max(xc),np.min(yc),np.max(yc)])
    cbar=plt.colorbar(plot1,ax=pl2)
    pl2.set_ylabel('gravity in mgal')
    #%%
    """Gravity of Topography and Ice"""
    
    thick=randomVar((x.shape[0],y.shape[0]),sigma=20).T*200+X*0.1-1000 # random thickness data
    thick=np.where(thick>0,thick,0) # state 0m Tickness with 0!
    thick= np.where(np.logical_and(thick+topo<0,thick!=0),0,thick)#prevent unlogic data
    #zerotopo=[]
    #for k in topoprisms: zerotopo.append(np.where(k!=np.inf,0,0))
    
    datdist=np.where(thick>0,2,0)#it is imortant to give the Datadistribution, so the algorithm knows where tho place the interfaces
    datdist=np.where(topo<0,1,datdist)
    #state 0 for ice-free, 1 for batthymetry and 2 for ice covered, floating ice is currently not supported
    topoprisms,iceprisms,densities=createMultResBodyfromTopo(topo,datdist,thick,suplevel=4) # prepare the Topographymodel
    
    grav=adaptiv_terrain(coords,topoprisms,iceprisms,densities,x,y,threshold=10**-4)
    
    _,(pl1,pl2,pl3)=plt.subplots(3)
    plot1=pl1.imshow(topo, extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
    pl1.set_ylabel('Bedrock Topography in m')
    cbar=plt.colorbar(plot1,ax=pl1)
    plot1=pl2.imshow(thick, extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
    pl2.set_ylabel('Ice thickness in m')
    pl1.scatter(coords[0,:],coords[1,:],s=3,marker='+',color=[1,1,1])
    cbar=plt.colorbar(plot1,ax=pl2)
    plot1=pl3.imshow(grav.reshape((coordx.shape[0],coordx.shape[1])),extent=[np.min(xc),np.max(xc),np.min(yc),np.max(yc)])
    pl3.set_ylabel('gravity in mgal')
    cbar=plt.colorbar(plot1,ax=pl3)
    #%%
    """Gravity of Moho""" 
    avg_moho_depth=25000
    moho=(topo*2670)/(2670-3200)-avg_moho_depth
    ref_lev=np.ones(moho.shape)*(-avg_moho_depth)#average depth of the moho
    prism=createrMultResBody(moho,ref_lev, suplevel=4)
    grav=adapt_Body(coords,prism,3200-2670,x,y,threshold=10**-4,maxrange=10_000)

    _,(pl1,pl2)=plt.subplots(2)
    
    plot1=pl1.imshow(moho, extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
    cbar=plt.colorbar(plot1,ax=pl1)
    pl1.set_ylabel('Mohoh height in m')
    pl1.scatter(coords[0,:],coords[1,:],s=3,marker='+',color=[1,1,1])
    #plt.imshow(topo, extent=[np.min(x),np.max(x),np.min(y),np.max(y)])
    plot1=pl2.imshow(grav.reshape((coordx.shape[0],coordx.shape[1])),extent=[np.min(xc),np.max(xc),np.min(yc),np.max(yc)])
    cbar=plt.colorbar(plot1,ax=pl2)
    pl2.set_ylabel('gravity in mgal')    