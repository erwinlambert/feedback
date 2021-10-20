import numpy as np

class Constants(object):
    """ Some general input
    
    """

    def __init__(self):
        
        self.spm = np.array([2678400,2419200,2678400,2592000,2678400,2592000,2678400,2678400,2592000,2678400,2592000,2678400]) #Seconds per month
        self.spy = 3600*24*365.25 #Seconds per year
        self.months = np.arange(0,12)
        
        self.K   = 1028*3947/(917*3.34e5) #Constant to convert gamma to basal melt sensitivity
        
        self.basin = ['East Ant.','Ross','Amundsen','Weddell','Peninsula','Pens. East','Pens. West']
        self.exp = ['eais','ross','amun','wedd','pens','tot2','hal2','doub']
        self.ssp = ['126','245','585']
        
        #Basal melt parameterisation
        self.gamma = {}
        self.gamma['lin']  = np.array([1.7e-5,4.8e-6,3.8e-5,9.6e-6,2.4e-5])
        self.gamma['quad'] = np.array([36e-5,36e-5,36e-5,36e-5,36e-5])
        
        self.bmpexp = {} #Exponent in basal melt equation
        self.bmpexp['lin'] = 1
        self.bmpexp['quad'] = 2
        
        #Perturbation magnitude
        self.pert = 400 #Gt/yr
        
        #Number of iterations in 
        self.niter = 1
        
        return