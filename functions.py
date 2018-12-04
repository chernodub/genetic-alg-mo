import matplotlib.pyplot as plt
import numpy as np

## each public function returns three variables 
# function: Function, func_dimension: int, function_range: [begin, end]*func_dimension

def __first(x):
    if (-1 > x[0] or x[0] > 1): return None
    econst = np.exp(x[0]**2 * (-2.77257))
    val = 0.05*(x[0] - 1)**2 + (3 - 2.9*econst) * (1 - np.cos(x[0] * (4 - 50*econst)))
    return val
def __printFirst():
    x = np.arange(-1, 1, 0.001)
    y = [__first([i]) for i in x]
    plt.plot(x, y)


def __third(x):
    x = np.array(x)
    if ((x > 16).any() or (x < -16).any()): return None
    val = 0.1*x[0]**2 + 0.1*x[1]**2 - 4*np.cos(0.8*x[0]) - 4*np.cos(0.8*x[1]) + 8
    return val
def __printThird():
    __printTwoDimensionalFunc(__third, [-16, 16])

def __fifth(x):
    x = np.array(x)
    if ((x > 2).any() or (x < -2).any()): return None
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
def __printFifth():
    __printTwoDimensionalFunc(__fifth, [-2, 2])

def __eighth(x):
    x = np.array(x)
    if ((x > 10).any() or (x < -10).any()): return None
    return (1 - np.sin(np.sqrt(x[0]**2 + x[1]**2))**2) / (1 + 0.001*(x[0]**2 + x[1]**2))
def __printEighth():
    __printTwoDimensionalFunc(__eighth, [-10, 10])
    
def __twelfth(x):
    x = np.array(x)
    if ((x > 4).any() or (x < 0).any()): return None
    val = 0.5*(x[0]**2 + x[0]*x[1] + x[1]**2)
    val = val / (1 + 0.5 * np.cos(1.5*x[0])*np.cos(3.2*x[0]*x[1])*np.cos(3.14*x[1]) +
                    0.5*np.cos(2.2*x[0])*np.cos(4.8*x[0]*x[1])*np.cos(3.5*x[1]))
    return val
def __printTwelfth():
    __printTwoDimensionalFunc(__twelfth, [0, 4])
    
def __printTwoDimensionalFunc(func, ranges):
    x = np.arange(ranges[0], ranges[1], 0.01)
    y = x[::-1]
    X,Y = np.meshgrid(x, y)
    Z = func([X, Y])
    im = plt.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], cmap='gist_heat') # drawing the function
    plt.colorbar(im) # adding the colobar on the right

def first():
    return __first, 1, [-1, 1], __printFirst

def third():
    return __third, 2, [-16, 16]*2, __printThird

def fifth():
    return __fifth, 2, [-2, 2]*2, __printFifth

def eighth():
    return __eighth, 2, [-10, 10]*2, __printEighth

def twelfth():
    return __twelfth, 2, [0, 4]*2, __printTwelfth
