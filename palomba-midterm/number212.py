import numpy as np

def loop(N):
        n = N + 1
        iter = 0
        while(n>N):
            N = n
            n = 32000*np.log(N)
            print(str(n))
            iter = iter + 1
            
loop(1000)