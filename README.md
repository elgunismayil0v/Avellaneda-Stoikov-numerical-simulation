
# Avellaneda-Stoikov-numerical-simulation
# Preparing the enviornment
### 	Required steps to run the program
	- Install > python 3.13.1 or above (for windows download from here https://www.python.org/downloads/release/python-3131/)
	- Create virtual env -> python -m venv venv
	- Activate it -> source venv/bin/activate (for windows command is just -> activate)
	- Get source code -> git clone https://github.com/elgunismayil0v/Avellaneda-Stoikov-numerical-simulation.git
	- Move to project folder -> cd Avellaneda-Stoikov-numerical-simulation
	- Install required libraries -> pip3 install -r requirements.txt or (pip install -r requirements.txt)
	- Run the program -> python -m src.main


## Class Diagram Overview:

- MarketSimulator (ABC)
  └─ ArithmeticBrownianMotion
  └─ GeometricBrownianMotion

- PricingStrategy (ABC)
  └─ AvellanedaStoikovStrategy
  └─ SymmetricStrategy

- OrderExecution (ABC)
  └─ PoissonOrderExecution

- InventoryManager
- DataLogger
- SimulationRunner

## The main components of the model are:

1. **Market Simulation**: Handles the mid-price dynamics, possibly using Brownian motion.

2. **Inventory Management**: Keeps track of the agent's current inventory and updates it based on trades.

3. **Pricing Strategy**: Computes reservation prices and optimal bid/ask spreads.

4. **Order Execution**: Manages how orders are placed and executed based on probabilities.

5. **Data Logging**: Records all the necessary data for analysis and visualization.

6. **Simulation Runner**: Coordinates the overall simulation, tying all components together.

We are using strategy design pattern, each of these components should be interchangeable. For example, different pricing strategies or market simulations can be plugged in without changing the rest of the code.

Starting with the Market Simulation. This is a base class that defines an interface for generating mid-prices. Subclasses can implement different models (e.g., Arithmetic Brownian Motion, Geometric Brownian Motion, etc.). The strategy pattern here allows different market models to be used interchangeably.

Next, the Pricing Strategy. This is a base class that defines methods to compute reservation prices and optimal spreads. The AvellanedaStoikovStrategy would be a concrete implementation based on the paper's formulas. Other strategies (like a symmetric strategy) can be other subclasses.

Order Execution handle how orders are processed. This is a base class that defines an interface for executing orders, with a PoissonOrderExecution class implementing the Poisson process described in the paper. This allows experimenting with different execution models.

Inventory Management is straightforward—it tracks the current inventory and cash. It contains methods to update when orders are filled.

Data Logging collect all relevant data points during the simulation. A base logger defines what data to collect, and different loggers (e.g., ConsoleLogger, FileLogger) can handle output.

The Simulation Runner ties everything together. It initializes the components, runs the simulation loop, and coordinates between the market, strategy, execution, etc.

## Development Workflow for 5-Person Team:
1. **Vikram Bahadur**: Quick POC, Base python classes blue print design and process implementation, team coordination and final presentation preparation.
2. **Elgun Ismayilov**: Extended MarketSimulator with new models (e.g.,gbm), Finding appropriate model to be implemented for whole team.
3. **Khalil Khalilli**: Implement alternative PricingStrategy classes, added unit test cases, graph plotting and utility optimization.
4. **Nihad Alili**: Parameters sensitive analysis and comparative strategies analysis.
5. **Anindita Basu**: Monte Carlo overall execution and results reporting with key results finding.




