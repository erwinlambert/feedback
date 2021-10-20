import numpy as np
import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 1.
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.subplot.wspace'] = .25
mpl.rcParams['figure.subplot.left'] = .1
mpl.rcParams['figure.subplot.right'] = .95
mpl.rcParams['figure.figsize'] = (7,5)

bcol = {}
bcol['East Ant.'] = 'tab:blue'
bcol['Ross']      = 'tab:orange'
bcol['Amundsen']  = 'tab:red'
bcol['Weddell']   = 'tab:purple'
bcol['Peninsula'] = 'tab:green'
bcol['Pens. East']= 'tab:green'
bcol['Pens. West']= 'tab:green'

scol = {}
scol['119'] = (0/255,173/255,207/255)
scol['126'] = (23/255,60/255,102/255)
scol['245'] = (247/255,148/255,32/255)
scol['370'] = (231/255,29/255,37/255)
scol['585'] = (149/255,27/255,30/255)
