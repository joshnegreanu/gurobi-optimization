import gurobipy as gp

# Check version
print(f"Gurobi Version: {gp.gurobi.version()}")

# Create a simple model to test license validity
try:
    m = gp.Model("test")
    m.optimize()
    print("License and Solver validation: SUCCESS")
except gp.GurobiError as e:
    print(f"Error: {e}")