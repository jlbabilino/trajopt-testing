import numpy as np
from sympy import *
from sympy.matrices.dense import hessian

def construct_polynomial(t, x0, x1, x2, x3, x4, x5):
  return x0 + x1*t + x2*t**2 + x3*t**3 + x4*t**4 + x5*t**5

def main():
  t = symbols("t")
  x0, x1, x2, x3, x4, x5 = symbols("x0, x1, x2, x3, x4, x5")
  y0, y1, y2, y3, y4, y5 = symbols("y0, y1, y2, y3, y4, y5")
  theta0, theta1, theta2, theta3, theta4, theta5 = symbols("θ0, θ1, θ2, θ3, θ4, θ5")

  x = construct_polynomial(t, x0, x1, x2, x3, x4, x5)
  y = construct_polynomial(t, y0, y1, y2, y3, y4, y5)
  theta = construct_polynomial(t, theta0, theta1, theta2, theta3, theta4, theta5)

  x_prime = diff(x, t, 1)
  y_prime = diff(y, t, 1)

  x_prime_prime = diff(x, t, 2)
  y_prime_prime = diff(y, t, 2)
  theta_prime_prime = diff(theta, t, 2)

  x_t0, y_t0, theta_t0 = symbols("x_t0 y_t0 θ_t0")
  x_t1, y_t1, theta_t1 = symbols("x_t1 y_t1 θ_t1")

  v_xt0, v_yt0 = symbols("v_xt0 v_yt0")
  v_xt1, v_yt1 = symbols("v_xt1 v_yt1")

  k_0, k_1 = symbols("k_0 k_1")

  c0 = Eq(x.subs(t, 0), x_t0)
  c1 = Eq(y.subs(t, 0), y_t0)
  c2 = Eq(theta.subs(t, 0), theta_t0)
  c3 = Eq(x.subs(t, 1), x_t1)
  c4 = Eq(y.subs(t, 1), y_t1)
  c5 = Eq(theta.subs(t, 1), theta_t1)
  c6 = Eq(x_prime.subs(t, 0), k_0 * v_xt0)
  c7 = Eq(y_prime.subs(t, 0), k_0 * v_yt0)
  c8 = Eq(x_prime.subs(t, 1), k_1 * v_xt1)
  c9 = Eq(y_prime.subs(t, 1), k_1 * v_yt1)

  spline_coeffs = (x0, x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, y5, theta0, theta1, theta2, theta3, theta4, theta5)

  solution, = linsolve([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9], *spline_coeffs)

  cost = integrate(x_prime_prime**2, (t, 0, 1)) + integrate(y_prime_prime**2, (t, 0, 1)) + integrate(theta_prime_prime**2, (t, 0, 1))
  
  cost_subs = cost
  for i in range(len(spline_coeffs)):
    cost_subs = cost_subs.subs(spline_coeffs[i], solution[i])

  # pprint(cost_subs)


  opti_params = (x4, x5, y4, y5, theta2, theta3, theta4, theta5, k_0, k_1)
  opti_params2 = [x4, x5, y4, y5, theta2, theta3, theta4, theta5, k_0, k_1]


  costhessian = hessian(cost_subs, opti_params)

  # costhessian_subs = costhessian
  # costhessian_subs = costhessian_subs.subs()

  # pprint(costhessian)
  b = Matrix([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

  mysol = linsolve((costhessian, b), opti_params2)

  pprint(mysol)



if __name__ == "__main__":
  main()
