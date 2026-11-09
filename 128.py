import math
N=100
X=8
def logf(p):
   if p <= 0 or p >= 1: return -math.inf
   return math.log(p)*X + math.log(1-p)*(N-X) # termenul C(N,X) nu depinde de p, nu va influenta semnul derivatei

def derivative_logf(p):
   if p <= 0: return math.inf
   if p >= 1: return -math.inf
   return X/p-(N-X)/(1-p)

def derivative2_logf(p):
   if p <= 0: return math.inf
   if p >= 1: return -math.inf
   return -X/(p*p)-(N-X)/((1-p)*(1-p))

def gradient_descent(stop_delta):
    MAX_LEARNING_RATE = 0.001
    p = 0.5
    iter=1
    while True:
        p_new = p+MAX_LEARNING_RATE*derivative_logf(p)
        print(p_new, iter)
        iter+=1
        if abs(p-p_new) <= stop_delta:
            return p
        p = p_new

def newton(stop_delta):
    MAX_LEARNING_RATE = 1
    p = 0.5
    iter = 1
    while True:
        if derivative2_logf(p)==0:
            return p
        p_new = p-MAX_LEARNING_RATE*(derivative_logf(p)/derivative2_logf(p))
        iter += 1
        if abs(p-p_new) <= stop_delta:
            return p
        p = p_new

print(f"Gradient descent: {gradient_descent(1e-4)}")
print(f"Newton's method: {newton(1e-4)}")
print(derivative_logf(0.5))