# AI

This repo tries to collect important knowledge about AI into a markdown documentation with hands-on examples and code.<br>
You might want to read the document or just view the code.<br>
In PyTorch there are many repatitive tasks and code, and this repo wants to help by providing templates and examples.



Table of Contents:
- [PyTorch](./docs/pytorch.md)
- [TensorFlow](./docs/tensorflow.md)
- [Scikit Learn](./docs/scikit_learn.md)
- [Optimization](#optimization)
- [Databases](#databases)
- [Ressources](#ressources)



Table of Code:
- ...



Organization:
- [Planning](#planning)



> Don't expect deep explanations, just shallow in-a-nutshell explanations. This repo is just a little helper.



<br><br>

---
### Optimization

[<img align="right" width=150px src='./res/rackete_2.png'></img>](#ai)

In principle, machine learning does nothing other than always find a maximum or usually a minimum, but actually always an optimum. For very difficult and unstructured problems, you use data and neural networks or possibly classic ML algorithms. Since in many cases, the problem cannot be broken down to a simple optimization problem at all or only to a very limited extent, and certainly cannot be calculated. ML algorithms use data to independently learn how to adjust their parameters/input in order to get closer to the global optimum in the end.

However, this is very expensive, requires vast amounts of data that is usually difficult to obtain and also needs a lot of computational resources. 

Many problems can also be solved with simple ML algorithms or other (non-data) learning algorithms. This help page offers a small overview of a few common optimization libraries in Python + Genetic Algorithm, which is very expensive, but also copes with very large parameter spaces.

I would like to introduce you to the tools **scipy.optimize**, **cvxpy** and **pyomo**. These Python libraries are ideal for solving optimization problems. I will show you a simple example to illustrate the functionalities.

<br><br>

1. **scipy.optimize**

    scipy.optimize is a library that provides various algorithms for numerical optimization. It is well suited for linear and non-linear optimization problems where the objective function and the constraints are defined as Python functions.

    Example: Linear optimization with `scipy.optimize.linprog`

    **Goal Function**: Minimize -> $C = 5x + 3y$ <br> 
    **Condition**:<br>
    $
    \begin{aligned}
    x + y &\geq 10 \\
    x &\geq 2 \\
    y &\geq 3
    \end{aligned}
    $

    Code:

    ```python
    import scipy.optimize as opt

    # Coefficients of the objective function
    c = [5, 3]  # Minimize 5x + 3y

    # Coefficients of the inequality constraints (Ax >= b)
    A = [[-1, -1], [-1, 0], [0, -1]]  # x + y >= 10, x >= 2, y >= 3
    b = [-10, -2, -3]

    # Solve the problem
    result = opt.linprog(c, A_ub=A, b_ub=b, method='simplex')

    print(result)
    ```

    Result:<br>
    The result will show you the values of \( x \) and \( y \) that minimize the objective function and satisfy the constraints at the same time.


<br><br>

2. **cvxpy**

    cvxpy is a Python library for convex optimization. It allows the formulation and solution of optimization problems that are convex and offers a very user-friendly interface.

    Example: Quadratic optimization with `cvxpy`

    **Goal Function**: Minimize -> $C = x^2 + y^2$ <br>  
    **Condition**:<br>
    $
    \begin{aligned}
    x + y &\geq 10 \\
    x &\geq 2 \\
    y &\geq 3
    \end{aligned}
    $

    ### Code:

    ```python
    import cvxpy as cp

    # Define variables
    x = cp.Variable()
    y = cp.Variable()

    # Define objective function (minimize x^2 + y^2)
    objective = cp.Minimize(x**2 + y**2)

    # Define constraints
    constraints = [x + y >= 10, x >= 2, y >= 3]

    # Define problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()

    print("Optimal x:", x.value)
    print("Optimal y:", y.value)
    ```

    Result:<br>
    The result will give you the values of \( x \) and \( y \) that minimize the objective function and simultaneously satisfy the constraints.

<br><br>

3. **pyomo**

    Pyomo is a very powerful Python library for modelling and solving optimization problems. It is suitable for both linear and non-linear optimization and offers extensive functions for complex optimization models.

    Example linear optimization with `pyomo`

    **Goalfunction**: Minimize -> $C = 5x + 3y$ <br> 
    **Condition**:<br>
    $
    \begin{aligned}
    x + y &\geq 10 \\
    x &\geq 2 \\
    y &\geq 3
    \end{aligned}
    $

    Code:

    ```python
    from pyomo.environ import *

    # Create model
    model = ConcreteModel()

    # Define variables
    model.x = Var(within=NonNegativeReals)
    model.y = Var(within=NonNegativeReals)

    # Define objective function
    model.obj = Objective(expr=5*model.x + 3*model.y, sense=minimize)

    # Define constraints
    model.con1 = Constraint(expr=model.x + model.y >= 10)
    model.con2 = Constraint(expr=model.x >= 2)
    model.con3 = Constraint(expr=model.y >= 3)

    # Solver
    solver = SolverFactory('glpk')
    solver.solve(model)

    print(f"Optimal x: {model.x.value}")
    print(f"Optimal y: {model.y.value}")
    ```

    Result:<br>
    The result shows you the optimum values for \( x \) and \( y \) that minimize the objective function and simultaneously satisfy the constraints.



4. **Genetic Algorithm**

    Genetic Algorithms (GAs) are inspired by the principles of natural selection and genetics. They are particularly useful for solving complex optimization problems where traditional methods struggle, especially when the search space is large, non-continuous, or not well-behaved.

    Example linear optimization using a **Genetic Algorithm** (with `deap` library)

    **Goalfunction**: Minimize -> $C = 5x + 3y$ <br>
    **Condition**:<br>
    $
    \begin{aligned}
    x + y &\geq 10 \\
    x &\geq 2 \\
    y &\geq 3
    \end{aligned}
    $

    Code:

    ```python
    import random
    from deap import base, creator, tools, algorithms

    # Define bounds for variables
    LOWER_BOUND = [2, 3]  # x >= 2, y >= 3
    UPPER_BOUND = [20, 20]  # Arbitrary upper bounds for search space

    # Fitness function
    def eval_cost(individual):
        x, y = individual
        # Penalty for violating constraints
        if x + y < 10:
            return 1e6,  # Large penalty
        return 5*x + 3*y,

    # Set up genetic algorithm
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimization
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda low, up: random.uniform(low, up))
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    [lambda: toolbox.attr_float(l, u) for l, u in zip(LOWER_BOUND, UPPER_BOUND)], n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_cost)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run GA
    random.seed(42)
    pop = toolbox.population(n=50)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=False)

    # Get best solution
    best_ind = tools.selBest(pop, 1)[0]
    print(f"Optimal x: {best_ind[0]:.2f}")
    print(f"Optimal y: {best_ind[1]:.2f}")
    print(f"Minimum cost: {eval_cost(best_ind)[0]:.2f}")
    ```

    Result:<br>
    This approach returns approximate solutions that respect the constraints and provide a good (but not necessarily exact) minimum of the cost function. It's especially helpful when the search space is too complex for analytical solvers.

    <br><br>

    Another example is the [Simple-Genetic-Algorithm](https://pypi.org/project/Simple-Genetic-Algorithm/) pip package, which is still in developement and improvement but still provides a good foundation usage of the gentic algorithm.

    ```python
    from genetic_algorithm import GA, get_random

    # Define GA class for linear optimization
    class Linear_GA(GA):
        def calculate_fitness(self, kwargs, params):
            x = params["x"]
            y = params["y"]

            # Constraints
            if (x + y < 10) or (x < 2) or (y < 3):
                return float('-inf')  # Infeasible solution

            # Objective function: minimize 5x + 3y => we maximize its negative
            return - (5 * x + 3 * y)

        def get_random_param_value(self, param_key):
            if param_key == "x":
                return get_random(0, 20)  # Adjust range as needed
            elif param_key == "y":
                return get_random(0, 20)

    # Define parameters to optimize
    parameters = {
        "x": 0,
        "y": 0
    }

    # Initialize optimizer
    optimizer = Linear_GA(
        generations=50,
        population_size=100,
        mutation_rate=0.2,
        list_of_params=parameters
    )

    # Optional: Add a few feasible starting points
    for x in range(2, 6):
        for y in range(3, 6):
            if x + y >= 10:
                optimizer.add_initial_solution({"x": x, "y": y})

    # Optimize
    best_params, best_fitness, log = optimizer.optimize()

    # Show result
    print(f"Best solution: x = {best_params['x']}, y = {best_params['y']}")
    print(f"Objective function value (minimized): {5 * best_params['x'] + 3 * best_params['y']}")
    ```


<br><br>

**Conclusion:**
- **scipy.optimize** is ideal for quick, simple optimization tasks.
- **cvxpy** is ideal for convex problems and offers an intuitive API.
- **pyomo** is the best choice for complex, extensive optimization models, even if the learning curve is somewhat steeper.
- **Genetic Algorithm** comes into play if the optimization problem is not just a function (like hyperparameter optimization) and the optimization space is huge with many parameters to optimize.



<br><br>

---
### Databases

[<img align="right" width=150px src='./res/rackete_2.png'></img>](#ai)

Coming soon...



<br><br>

---
### Ressources

[<img align="right" width=150px src='./res/rackete_2.png'></img>](#ai)

There is super much ressource for learning ai. Sadly many courses try to benefit from the ai hype and so you should inform yourself before buying a book or a course, some learning content covers super much content but without explanations, which is in most cases not helpful for learning.<br>
It follows a list of some important/helpful ressources.

- [PyTorch Udemy Course](https://www.udemy.com/course/pytorch-for-deep-learning/) -> on udemy and youtube there are much more
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- Deep Learning - Das umfassende Handbuch -> [amazon](https://www.amazon.de/Deep-Learning-umfassende-Handbuch-Forschungsans%C3%A4tze/dp/3958457002), [heise](https://shop.heise.de/deep-learning-das-umfassende-handbuch), [thalia](https://www.thalia.de/shop/home/artikeldetails/A1052910632), [content-select](https://content-select.com/de/portal/media/view/5c858648-0080-4ea3-be06-6037b0dd2d03)
- and there are much more good books like [Deep Learning from Ian Goodfellow](https://www.amazon.de/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618)

-> add more ressources


> I by myself learned most of ai in my university which covered general informatics, maths, classic machine learning with scikit learn, visualization with matplotlib, computer vision with cv2, neural networks with pytorch, databases with SQL, data engineering and much more.



<br><br>

---
### Planning

[<img align="right" width=150px src='./res/rackete_2.png'></img>](#ai)

Ideas for code and explanation:
- Scikit (Classic Machine Learning)
- PyTorch
- TensorFlow (?)

- AI Core
    - Neuronal Networks
        - Definition
        - What it really does
        - What needed for a nn? (all components, puzzle parts)
        - code
    - Foundation Models
- Neuronal Networks
    - Architectures
        - Fully Connected
        - CNN
        - ResNet
        - ConvNeXt
        - U-Net
        - RNN
        - LSTM
        - Transformer
    - Learning Methods
        - Supervised
        - Self-Supervised (DINO)
        - Embedding Learning CLIP/Autoencoder
        - Reinforcement Learning
        - Generative Modelle
        - (Unsupervised)
    - Losses
    - Activation Functions
    - Regularizations (Batch/Layer-Normalization, ...)
- Tools /AI-Related
    - Git
    - Anaconda
    - Docker
    - Remote Access
    - Configuration
    - Optimization
- Foundations
    - Math
    - Python / Programming
    - Databases?
- Others
    - Ethics? (What is Allowed to do? How have 












