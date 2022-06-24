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
        self.exp = ['eais','ross','amun','wedd','pens','tot2','hal2']
        self.ssp = ['126','245','585']
        
        self.bmps = ['lin','quad']
        
        #Basal melt parameterisation
        self.gamma0 = 2.5 #Initial value for both linear and quadratic params
        
        self.bmpexp = 2 #Quadratic, can be changed to 1 for linear param
        
        #Perturbation magnitude
        self.pert = 400 #Gt/yr
        
        #Number of iterations
        self.niter = 40
        
        self.Tf = -1.7
        
        self.Trean = np.array([0.53,-0.18,1.37,-0.79,-0.24]) #Reanalysis values

        self.SLRtarget = .0132 #meters over 40 year Rignot period | Antarctic: .0132; Amundsen: .0096
        
        self.bcol = {}
        self.bcol['East Ant.'] = 'tab:blue'
        self.bcol['Ross']      = 'tab:orange'
        self.bcol['Amundsen']  = 'tab:red'
        self.bcol['Weddell']   = 'tab:purple'
        self.bcol['Peninsula'] = 'tab:green'
        self.bcol['Pens. East']= 'tab:green'
        self.bcol['Pens. West']= 'tab:green'

        self.scol = {}
        self.scol['119'] = (0/255,173/255,207/255)
        self.scol['126'] = (23/255,60/255,102/255)
        self.scol['245'] = (247/255,148/255,32/255)
        self.scol['370'] = (231/255,29/255,37/255)
        self.scol['585'] = (149/255,27/255,30/255)

        self.rcol = {}
        self.rcol['ctrl'] = '.5'
        self.rcol['quar'] = 'tab:cyan'
        self.rcol['half'] = 'tab:olive'
        self.rcol['hal2'] = 'tab:olive'
        self.rcol['totl'] = 'tab:pink'
        self.rcol['tot2'] = 'tab:pink'
        self.rcol['doub'] = 'tab:brown'
        self.rcol['eais'] = self.bcol['East Ant.']
        self.rcol['ross'] = self.bcol['Ross']
        self.rcol['amun'] = self.bcol['Amundsen']
        self.rcol['wedd'] = self.bcol['Weddell']
        self.rcol['pens'] = self.bcol['Peninsula']
        
        return