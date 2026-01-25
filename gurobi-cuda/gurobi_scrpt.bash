uname -m
# Output MUST be: x86_64

python3 --version
# Output MUST be: Python 3.14.x

# 1. Create a virtual environment named 'gurobi_env'
python3.14 -m venv gurobi_env

# 2. Activate the environment
source gurobi_env/bin/activate

# Upgrade pip to ensure it can parse the latest wheel tags
pip install --upgrade pip

# Install the specific wheel file
# Replace /path/to/ with the actual location of your file
pip install /path/to/gurobipy-13.0.1+cu131-cp314-cp314-manylinux_2_24_x86_64.whl

mv gurobipy-13.0.1+cu131-cp314-cp314-manylinux_2_24_x86_64.whl gurobipy-13.0.1+cu131-cp314-cp314-linux_x86_64.whl
pip install gurobipy-13.0.1+cu131-cp314-cp314-linux_x86_64.whl