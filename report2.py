import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def util(cons,gamma):
  return max(cons, 0.0001)**(1.0-gamma)/(1.0-gamma)

#parameters
gamma = 2.0
beta = 0.985**20
r = 1.025**20-1.0
jj = 60
l = np.array([0.8027, 1.0, 1.2457])
NL = 3
prob = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1361],
    [0.0021, 0.2528, 0.7451]
])
mu_1 = np.array([1.0/NL,1.0/NL,1.0/NL])

mu_2 = np.zeros(NL)

for il in range(NL):
    for ilp in range(NL):
        mu_2[ilp] += prob[il,ilp]*mu_1[il]

# grids
a_l = 0.0
a_u = 10.0
NA = 100
a = np.linspace(a_l, a_u, NA)

# initialization
v = np.zeros((jj, NA, NL))
iaplus = np.zeros((jj, NA, NL), dtype=int)
aplus = np.zeros((jj, NA, NL))
c = np.zeros((jj, NA, NL))

middle_age = range(20, 40)
tax_rate = 0.3
tax_revenue = 0.0
for ij in middle_age:
  for il in range(NL):
    for ia in range(NA):
      tax_revenue += tax_rate * l[il] * mu_2[il]

print("政府の総収入",tax_revenue * len(middle_age))

tax_total = tax_revenue * len(middle_age)
tax_return = tax_total * (1 + r)
old_age = range(40, 60)
old_period = len(old_age)

print("1期あたりの年金受給額", tax_return / old_period)