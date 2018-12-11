import matplotlib.pyplot as plt
import numpy as np

## each public function returns three variables 
# function - the math function iself          
# dimension - integer dimension of function                              
# range - the range of function values ([1,2] if n=1, [1,2,1,2] if n=2)  
# getPlotFunction - function that builds the plot

def __first(x):
    if (-1 > x[0] or x[0] > 1): return None
    econst = np.exp(x[0]**2 * (-2.77257))
    val = 0.05*(x[0] - 1)**2 + (3 - 2.9*econst) * (1 - np.cos(x[0] * (4 - 50*econst)))
    return val
def __getPlotFirst(bestPoint=[0]):
    x = np.arange(-1, 1, 0.001)
    y = [__first([i]) for i in x]
    fig, (base, zoomed) = plt.subplots(nrows=2)
    baseImage = base.plot(x, y)

    x = np.arange(bestPoint[0] - 2*0.1, bestPoint[0] + 2*0.1, 0.001)
    y = [__first([i]) for i in x]
    zoomedImage = zoomed.plot(x, y)
    
    return base, zoomed, [bestPoint[0] - 2*0.1, bestPoint[0] + 2*0.1]

def __third(x):
    x = np.array(x)
    if ((x > 16).any() or (x < -16).any()): return None
    val = 0.1*x[0]**2 + 0.1*x[1]**2 - 4*np.cos(0.8*x[0]) - 4*np.cos(0.8*x[1]) + 8
    return val
def __getPlotThird(bestPoint=[0, 0]):
    return __getPlotTwoDimensionalFunc(__third, [-16, 16]*2, bestPoint)

def __fifth(x):
    x = np.array(x)
    if ((x > 2).any() or (x < -2).any()): return None
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
def __getPlotFifth(bestPoint=[0, 0]):
    return __getPlotTwoDimensionalFunc(__fifth, [-2, 2]*2, bestPoint)

def __eighth(x):
    x = np.array(x)
    if ((x > 10).any() or (x < -10).any()): return None
    return (1 - np.sin(np.sqrt(x[0]**2 + x[1]**2))**2) / (1 + 0.001*(x[0]**2 + x[1]**2))
def __getPlotEighth(bestPoint=[0, 0]):
    return __getPlotTwoDimensionalFunc(__eighth, [-10, 10]*2, bestPoint)
    
def __twelfth(x):
    x = np.array(x)
    if ((x > 4).any() or (x < 0).any()): return None
    val = 0.5*(x[0]**2 + x[0]*x[1] + x[1]**2)
    val = val / (1 + 0.5 * np.cos(1.5*x[0])*np.cos(3.2*x[0]*x[1])*np.cos(3.14*x[1]) +
                    0.5*np.cos(2.2*x[0])*np.cos(4.8*x[0]*x[1])*np.cos(3.5*x[1]))
    return val
def __getPlotTwelfth(bestPoint=[0, 0]):
    return __getPlotTwoDimensionalFunc(__twelfth, [0, 4]*2, bestPoint)
    
def __getPlotTwoDimensionalFunc(func, ranges, bestPoint):
    step = 0.01
    x = np.arange(ranges[0], ranges[1], step)
    y = np.arange(ranges[2], ranges[3], step)[::-1]
    X,Y = np.meshgrid(x, y)
    Z = func([X, Y])
    fig, (base, zoomed) = plt.subplots(ncols=2)
    base.set_title("Base plot")
    baseImage = base.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], cmap='gist_heat') # drawing the function
    
    scope_coef = 0.05

    left_range = bestPoint[0] - abs(ranges[0] - ranges[1])*scope_coef
    right_range = bestPoint[0] + abs(ranges[0] - ranges[1])*scope_coef
    bottom_range = bestPoint[1] - abs(ranges[2] - ranges[3])*scope_coef
    top_range = bestPoint[1] + abs(ranges[2] - ranges[3])*scope_coef

    if (left_range < ranges[0]): left_range = ranges[0]
    if (right_range > ranges[1]): right_range = ranges[1]
    if (bottom_range < ranges[2]): bottom_range = ranges[2]
    if (top_range > ranges[3]): top_range = ranges[3]

    x = np.arange(left_range, right_range, step)
    y = np.arange(bottom_range, top_range, step)[::-1]
    X,Y = np.meshgrid(x, y)
    Z = func([X, Y])
    zoomedImage = zoomed.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], cmap='gist_heat')
    zoomed.set_title("Zoomed plot")
    fig.colorbar(baseImage, ax=base) # adding the colobar on the right
    fig.colorbar(zoomedImage, ax=zoomed)
    return base, zoomed, [left_range, right_range, bottom_range, top_range]

def first():
    return __first, 1, [-1, 1], __getPlotFirst

def third():
    return __third, 2, [-16, 16]*2, __getPlotThird

def fifth():
    return __fifth, 2, [-2, 2]*2, __getPlotFifth

def eighth():
    return __eighth, 2, [-10, 10]*2, __getPlotEighth

def twelfth():
    return __twelfth, 2, [0, 4]*2, __getPlotTwelfth


