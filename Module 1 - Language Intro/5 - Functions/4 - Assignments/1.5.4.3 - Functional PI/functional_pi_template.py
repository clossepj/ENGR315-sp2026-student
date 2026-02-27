import math


def my_pi(target_error):
    
    


    """
    Implementation of Gauss–Legendre algorithm to approximate PI from https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_algorithm

    :param target_error: Desired error for PI estimation
    :return: Approximation of PI to specified error bound
    """

    ### YOUR CODE HERE ###

    a = 1
    b = 1 / math.sqrt(2)
    t = 1 / 4
    p = 1  
    
    while True:
     a1 = (a + b) / 2
     b1 = math.sqrt(a*b)
     t1 = t - (p * (a1 - a) ** 2)
     p1 = 2 * p

     a = a1
     b = b1
     t = t1
     p = p1
        
     pi_estimate = ((a + b) ** 2) / (4 * t)
     
     if abs(math.pi - pi_estimate) < target_error:
        return pi_estimate
    
    # change this so an actual value is returned
    




desired_error = 1E-10

approximation = my_pi(desired_error)

print("Solution returned PI=", approximation)

error = abs(math.pi - approximation)

if error < abs(desired_error):
    print("Solution is acceptable")
else:
    print("Solution is not acceptable")
