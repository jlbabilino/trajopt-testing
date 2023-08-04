import numpy as np
from sympy import *
from sympy.matrices.dense import hessian
from dataclasses import dataclass

def construct_polynomial(t: Symbol, coeffs: list[Symbol]) -> Symbol:
  polynomial = 0
  for i in range(len(coeffs)):
    polynomial += coeffs[i] * t ** i

  return polynomial

def create_indexed_symbols(prefix: str, indices: list[int]) -> list[Symbol]:
  new_symbols = []
  for index in indices:
    new_symbols.append(symbols(prefix + "i" + str(index)))
  return new_symbols

def main():
  generate_optimal_spline(5, [
    Waypoint(1, 2, 0),
    Waypoint(5, 4, 0),
    Waypoint(1, 7, 0),
    Waypoint(3, 6, 0),
    Waypoint(5, 0, 0),
    Waypoint(5, 5, 0),
    Waypoint(2, 4, 0),
    Waypoint(8, 1, 0),
  ])
  # print(latex(myspline))
  

# @dataclass
class Waypoint:
  x: float
  y: float
  theta: float

  def __init__(self, x, y, theta):
    self.x = x
    self.y = y
    self.theta = theta

  def __str__(self):
    return "\\left(" + str(self.x) + "," + str(self.y) + "\\right)"

def generate_optimal_spline(degree: int, waypoints: list[Waypoint]):
  wpt_cnt = len(waypoints)
  sgmt_cnt = len(waypoints) - 1

  t = symbols("t")
  x = []
  y = []
  theta = []
  
  xCoeffs = []
  yCoeffs = []
  thetaCoeffs = []

  for sgmt_idx in range(sgmt_cnt):
    xCoeffs.append(create_indexed_symbols("x_s" + str(sgmt_idx), range(degree + 1)))
    yCoeffs.append(create_indexed_symbols("y_s" + str(sgmt_idx), range(degree + 1)))
    thetaCoeffs.append(create_indexed_symbols("θ_s" + str(sgmt_idx), range(degree + 1)))
    x.append(construct_polynomial(t, xCoeffs[-1]))
    y.append(construct_polynomial(t, yCoeffs[-1]))
    theta.append(construct_polynomial(t, thetaCoeffs[-1]))

  constraints = []

  # Apply waypoint pose constraints
  for sgmt_idx in range(sgmt_cnt):
    constraints.append(Eq(x[sgmt_idx].subs(t, 0), waypoints[sgmt_idx].x))
    constraints.append(Eq(x[sgmt_idx].subs(t, 1), waypoints[sgmt_idx + 1].x))
    constraints.append(Eq(y[sgmt_idx].subs(t, 0), waypoints[sgmt_idx].y))
    constraints.append(Eq(y[sgmt_idx].subs(t, 1), waypoints[sgmt_idx + 1].y))
    constraints.append(Eq(theta[sgmt_idx].subs(t, 0), waypoints[sgmt_idx].theta))
    constraints.append(Eq(theta[sgmt_idx].subs(t, 1), waypoints[sgmt_idx + 1].theta))

  # Make multiple orders of derivatives equivalent at barrier between splines
  for deriv_order in range(1, 3):
    for wpt_idx in range(1, wpt_cnt - 1):
      constraints.append(Eq(diff(x[wpt_idx - 1], (t, deriv_order)).subs(t, 1), diff(x[wpt_idx], (t, deriv_order)).subs(t, 0)))
      constraints.append(Eq(diff(y[wpt_idx - 1], (t, deriv_order)).subs(t, 1), diff(y[wpt_idx], (t, deriv_order)).subs(t, 0)))
      constraints.append(Eq(diff(theta[wpt_idx - 1], (t, deriv_order)).subs(t, 1), diff(theta[wpt_idx], (t, deriv_order)).subs(t, 0)))

  spline_coeffs = []

  for sgmt_idx in range(sgmt_cnt):
    for coeffs in (xCoeffs, yCoeffs, thetaCoeffs):
      for coeff in coeffs[sgmt_idx]:
        spline_coeffs.append(coeff)

  solution, = linsolve(constraints, *spline_coeffs)

  cost = 0
  for vari in (x, y, theta):
    for sgmt_idx in range(sgmt_cnt):
      cost += integrate(diff(vari[sgmt_idx], (t, 2))**2, (t, 0, 1))

  for i in range(len(spline_coeffs)):
    cost = cost.subs(spline_coeffs[i], solution[i])

  free_variables = list(cost.free_symbols)

  cost_hessian = hessian(cost, free_variables)

  b = Matrix([0] * len(free_variables))

  final_sol, = linsolve((cost_hessian, b), free_variables)

  solution = solution.subs(list(zip(free_variables, final_sol)))

  for sgmt_idx in range(sgmt_cnt):
    x[sgmt_idx] = x[sgmt_idx].subs(list(zip(spline_coeffs, solution)))
    y[sgmt_idx] = y[sgmt_idx].subs(list(zip(spline_coeffs, solution)))
    theta[sgmt_idx] = theta[sgmt_idx].subs(list(zip(spline_coeffs, solution)))
  
  splines = []
  for sgmt_idx in range(sgmt_cnt):
    splines.append((x[sgmt_idx], y[sgmt_idx]))
    pprint(latex(splines[-1]))

  for wpt in waypoints:
    print(wpt)

if __name__ == "__main__":
  main()
