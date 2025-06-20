# AI

This repo tries to collect important knowledge about AI into a markdown documentation with hands-on examples and code.<br>
You might want to read the document or just view the code.<br>
In PyTorch there are many repatitive tasks and code, and this repo wants to help by providing templates and examples.



Table of Contents:
- [PyTorch](./docs/pytorch.md)
- [TensorFlow](./docs/tensorflow.md)
- [Scikit Learn](./docs/scikit_learn.md)
- [Optimization](#optimization)
- [ONNX](#onnx)
- [Experiment Tracking](#experiment-tracking)
- [Visualization](#visualization)
- [Using a already made model](#using-a-already-made-model)
- [Databases](#databases)
- [Ressources](#ressources)


Table of Code:

- **Numpy**
    - [Basics](./src/examples/numpy/basics.ipynb)
    - [Basics (task based)](./src/examples/numpy/basics_task_based.ipynb)
    - [Generation](./src/examples/numpy/generation.ipynb)
    - [Sclicing](./src/examples/numpy/slicing.ipynb)
    - [Slicing and Indexing](./src/examples/numpy/slicing_and_indexing.ipynb)
    - [Advanced Stuff](./src/examples/numpy/advanced_stuff.ipynb)
    - [Wiederholung und Vektorisierung (Performance test)](./src/examples/numpy/Wiederholung%20und%20Vektorisierung.ipynb)
- **Pandas**
    - [Series](./src/examples/pandas/series.ipynb)
    - [DataFrame](./src/examples/pandas/dataframe.ipynb)
    - [DataFrame (other version)](./src/examples/pandas/dataframe_another_version.ipynb)
    - [I/O](./src/examples/pandas/io.ipynb)
    - [Missing Data](./src/examples/pandas/missing_data.ipynb)
- **Data Engineering**
    - [Data Wrangling](./src/examples/data_engineering/Data_Wrangling.ipynb)
    - [Data Collection](./src/examples/data_engineering/Data_Collection.ipynb)
    - [Data Scraping](./src/examples/data_engineering/Data_Scraping.ipynb)
    - [Data Generation](./src/examples/data_engineering/data_generation.ipynb)
    - [Generators](./src/examples/data_engineering/generators.ipynb)
    - [Group By](./src/examples/data_engineering/GroupBy.ipynb)
    - [Pandas Group by](./src/examples/data_engineering/pandas_Group_by.ipynb)
    - [Pandas Merge and Join](./src/examples/data_engineering/pandas_Merge_and_Join.ipynb)
    - [Pandas Reshape](./src/examples/data_engineering/pandas_reshape.ipynb)
    - [DASK](./src/examples/data_engineering/DASK.ipynb)
    - [Lambda Operators](./src/examples/data_engineering/Lambda_Operators.ipynb)
    - [Map Reduce](./src/examples/data_engineering/map_reduce.ipynb)
    - [Parallel Map Reduce](./src/examples/data_engineering/Parallel_Map_Reduce.ipynb)
    - [MRJOBLIB](./src/examples/data_engineering/MRJOBLIB.ipynb)
    - [HDF5](./src/examples/data_engineering/HDF5_intro.ipynb)
    - [Hive](./src/examples/data_engineering/Hive.ipynb)
    - [HBASE](./src/examples/data_engineering/HBASE.ipynb)
    - [NoSQL Mongo](./src/examples/data_engineering/NoSQL_Mongo.ipynb)
    - [Diagrams](./src/examples/data_engineering/Diagrams.ipynb)
    - [Rapids colab cuML demo](./src/examples/data_engineering/Lecture_rapids_colab_cuML_demo.ipynb)
    - [RAY Actors](./src/examples/data_engineering/Ray-Actors.ipynb)
    - [RAY Tasks](./src/examples/data_engineering/Ray-Tasks.ipynb)
    - [RAY Tune with MNIST](./src/examples/data_engineering/Ray-Tune-with-MNIST.ipynb)
    - [Numba demo](./src/examples/data_engineering/Numba_demo.ipynb)
- [**Computer Vision Examples**](https://github.com/xXAI-botXx/computer-vision)
- **Matplotlib**
    - [Visualisierung Teil 1](./src/examples/matplotlib/Visualisierungen%20Teil%201.ipynb)
    - [Visualisierung Teil 2](./src/examples/matplotlib/Visualisierungen%20Teil%202.ipynb)
    - [Basics](./src/examples/matplotlib/basics.ipynb)
    - [OOP Way](./src/examples/matplotlib/OOP_way.ipynb)
    - [Styling](./src/examples/matplotlib/styling.ipynb)
    - [Subplots](./src/examples/matplotlib/subplots.ipynb)
    - [Advanced Stuff](./src/examples/matplotlib/advanced_stuff.ipynb)
- **Analysis**
    - [UseCase NY Taxi](./src/examples/data_engineering/UseCase_NY_Taxi.ipynb)
    - [Feature Extraction](./src/examples/analysis/Feature_Extraction_Tobia.ipynb)
    <!--- [Example Data Analysis](./src/examples/analysis/Data_Analysis_Conveyor_Belt_Tobia.ipynb)-->
- **Sci-Kit Learn** (Sklearn)
    - [Basics](./src/examples/scikit-learn/basics.ipynb)
    - [Einführung Scikit-Learn](./src/examples/scikit-learn/Einführung_Scikit_Learn.ipynb)
    - [Data Transformation with sklearn](./src/examples/scikit-learn/Data%20Transformation%20with%20sklearn.ipynb)
    - [Feature Selection](./src/examples/scikit-learn/Feature%20Selection.ipynb)
    - [Decision Tree in sklearn](./src/examples/scikit-learn/Decision%20Tree%20in%20sklearn.ipynb)
    - [CIFAR10](./src/examples/scikit-learn/CIFAR10-ShallowLearning.ipynb)
    - [Heart attack](./src/examples/scikit-learn/Heart%20attack.ipynb)
    - [Voting Classifier](./src/examples/scikit-learn/Voting%20Classifier.ipynb)
    - [Gradientenabstieg](./src/examples/scikit-learn/Gradientenabstieg.ipynb)
    - [Dim-Reduktion](./src/examples/scikit-learn/Dim-Reduktion.ipynb)
    - [Density-based](./src/examples/scikit-learn/Density-based.ipynb)
    - [K-Means](./src/examples/scikit-learn/kmeans.ipynb)
    - [Linkage](./src/examples/scikit-learn/Linkage.ipynb)
    - [OPTICS](./src/examples/scikit-learn/OPTICS.ipynb)
    - [Association Mining](./src/examples/scikit-learn/Association%20Mining.ipynb)
    - [Indexstrukturen](./src/examples/scikit-learn/Indexstrukturen.ipynb)
    - [Zeitreihen Clustering](./src/examples/scikit-learn/Zeitreihen%20Clustering.ipynb)
- **NN**
    - [A simple Perceptron in NumPy](./src/examples/nn/A_simple_Perceptron_in_NumPy.ipynb)
    - [manual simple Perceptron in NumPy](./src/examples/nn/manual_simple_Perceptron_in_NumPy.ipynb)
    - [Multi Class Perceptrons](./src/examples/nn/Multi_Class_Perceptrons.ipynb)
- **PyTorch**
    - [Autograd Tutorial](./src/examples/pytorch/autograd_tutorial.ipynb)
    - [Using Tensorboard](./src/examples/pytorch/tensorboard_with_pytorch.ipynb)
    - [Tensors](./src/examples/pytorch/pytorch_tensors.ipynb)
    - [Tensors (another version)](./src/examples/pytorch/tensors.ipynb)
    - [Dataloader](./src/examples/pytorch/data_loader.ipynb)
    - [Data Transforms](./src/examples/pytorch/data_transforms.ipynb)
    - [Perceptron](./src/examples/pytorch/perceptron.ipynb)
    - [Basic MLP in Pytorch](./src/examples/pytorch/Basic_MLP_in_Pytorch.ipynb)
    - [Model Zoo](./src/examples/pytorch/Model_Zoo.ipynb)
    - [TSMixer (Regresion MLP)](./src/examples/pytorch/tsmixer.py)
    - [Model Evaluation TSMixer](./src/examples/pytorch/model_evaluation_TSMixer.ipynb)
    - [Stock Price Prediction](./src/examples/pytorch/stock_price_prediction.ipynb)
    - [Time-Series Prediction with LSTM](./src/examples/pytorch/Time_Series_Prediction_with_LSTM.ipynb)
    - [CIFAR10 MLP](./src/examples/pytorch/CIFAR10_MLP.ipynb)
    - [CIFAR10 MLP optimization](./src/examples/pytorch/CIFAR10_MLP_optimization.ipynb)
    - [CIFAR10 CNN](./src/examples/pytorch/CIFAR10_CNN.ipynb)
    - [ResNet (not finish)](./src/examples/resnet.ipynb)
    - [ResNet as Python file](./src/examples/resnet.py)
    - [ConvNext](./src/examples/ConvNext.ipynb)
    - [Dino with insides](./src/examples/dino_inside.ipynb)
    - [Clip and SWIN](./src/examples/CLIP_and_SWIN.ipynb)
    - [UNet](./src/examples/Unet.ipynb)
    - [MaskRCNN - Instance Segmentation](./src/examples/maskrcnn_toolkit.py) -> [see here if you want to know, how to use this py](https://github.com/xXAI-botXx/torch-mask-rcnn-instance-segmentation)
    - [Finetuning pretrained Transformer](./src/examples/pytorch/fine_tune_pre_trained_transformer.ipynb)
    - [Pretrained Vision Transformer](./src/examples/pytorch/pre_trained_vt.ipynb)
    - [GPT2 Inference](./src/examples/pytorch/GPT2_inference.ipynb)
    - [GPT2 Train](./src/examples/pytorch/GPT2_train.ipynb)
    - [GPT2 Train Refinetuning](./src/examples/pytorch/GPT2_train_refinetuning.ipynb)
    - [Transformer Tutorial](./src/examples/pytorch/transformer_tutorial.ipynb)
    - [DCGAN on CIFAR10](./src/examples/pytorch/DCGAN_CIFAR10.ipynb)
    - [GNN](./src/examples/pytorch/GNN_intro.ipynb)
    - [Graph Classification](./src/examples/pytorch/Graph_Classification.ipynb)
- **TensorFlow**
    - [Basics](./src/examples/tensorflow/basics.ipynb)
    - [Classification](./src/examples/tensorflow/classification.ipynb)
    - [Regression](./src/examples/tensorflow/regression.ipynb)
    - [MLP CIFAR10](./src/examples/tensorflow/MLP_CIFAR10.ipynb)
    - [CNN MNIST](./src/examples/tensorflow/cnn_mnist_data.ipynb)
    - [CNN CIFAR10](./src/examples/tensorflow/CNN_CIFAR10.ipynb)
    - [CNN CIFAR10 (another version)](./src/examples/tensorflow/cnn_cifar10_data.ipynb)
    - [Autoencoder Fashion-MNIST](./src/examples/tensorflow/Autoencoder_Fashion_MNIST.ipynb)
    - [Variational-Autoencoder CelebA](./src/examples/tensorflow/Variational_Autoencoder_CelebA.ipynb)
    - [DCGAN](./src/examples/tensorflow/DCGAN.ipynb)
    - [WGAN-GP](./src/examples/tensorflow/WGAN-GP.ipynb)
    - [Classic Q-Learning](./src/examples/tensorflow/classic_q_learning.ipynb)
    - [Reinforcement Learning](./src/examples/tensorflow/reinforcement_learning.ipynb)
    - [Deep Q Learning with only images](./src/examples/tensorflow/deep_q_learning_with_images.ipynb)
- **NLP**
    - [Text Klassifikation](./src/examples/nlp/Text-Klassifikation.ipynb)
    - [Introduction to spaCy](./src/examples/nlp/Introduction%20to%20spaCy.ipynb)
    - [Textvorbereitung](./src/examples/nlp/Textvorbereitung.ipynb)
    - [POS-Tagging](./src/examples/nlp/POS-Tagging.ipynb)
    - [Dependency Parsing](./src/examples/nlp/Dependency%20Parsing.ipynb)
    - [word2vec-Job Posts Similarity](./src/examples/nlp/word2vec-job-posts-similarity.ipynb)
    - [Text Generation](./src/examples/nlp/Text-Generation.ipynb)
- **ML-Systems**
    - [Git Tutorial](./src/examples/ml_systems/git_tutorial.ipynb)
    - [DVC Tutorial](./src/examples/ml_systems/dvc_tutorial.ipynb)
    - [MLFlow Tutorial](./src/examples/ml_systems/MLFlow_Tutorial.ipynb)
    - [MLFlow Example](./src/examples/ml_systems/MLFlow_example.ipynb)
    - [Autosklearn](./src/examples/ml_systems/autosklearn.ipynb)
    - [Evotorch MNIST30K](./src/examples/ml_systems/evotorch_MNIST30K.ipynb)
    - [DeepLake Tutorial](./src/examples/ml_systems/DeepLakeTutorial.ipynb)
    - [DeepLake PyTorch Example](./src/examples/ml_systems/DeepLakePyTorchExample.ipynb)
    - [Evidently](./src/examples/ml_systems/Evidently.ipynb)
    - [Voila Tutorial](./src/examples/ml_systems/voila_tutorial.ipynb)
    - [InteractiveML Regression](./src/examples/ml_systems/InteractiveML-Regression.ipynb)
    - [Jupyter Widgets](./src/examples/ml_systems/jupyter_widgets.ipynb)
    - [ONNX](./src/examples/ml_systems/ONNX.ipynb)
    - [CUDA Intro](./src/examples/ml_systems/CUDA_Intro.ipynb)
    - [Pip Tutorial](./src/examples/ml_systems/pip_tutorial.ipynb)
    - [Torch Package](./src/examples/ml_systems/torch_package.ipynb)
    - [Getting Started with PyTorch on Cloud TPUs](./src/examples/ml_systems/Getting_Started_with_PyTorch_on_Cloud_TPUs.ipynb)
    - [PyTorch TPU MNIST Training](./src/examples/ml_systems/PyTorch_TPU_MNIST_Training.ipynb)
    - [PyTorch TPU ResNet50 Inference](./src/examples/ml_systems/PyTorch_TPU_ResNet50_Inference.ipynb)
    - [NNI nas](./src/examples/ml_systems/NNI_nas.ipynb)
    - [Tunneling Tutorial](./src/examples/ml_systems/tunneling_tutorial.ipynb)
    - [PyTorch XLA Profling Colab Tutorial](./src/examples/ml_systems/PyTorch_XLA_Profling_Colab_Tutorial.ipynb)
    - [Intro to Weights & Biases](./src/examples/ml_systems/Intro_to_Weights_&_Biases.ipynb)
    - [Organizing Hyperparameter Sweeps in PyTorch with W&B](./src/examples/ml_systems/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb)
- **Helper**
    - [Image Plotting](./src/helper/imshow.py) -> [or use it directly from the pip packages prime_printer](https://github.com/xXAI-botXx/prime_printer)
    - [Hardware Check](./src/helper/hardware_check.py) -> [or use it directly from the pip packages prime_printer](https://github.com/xXAI-botXx/prime_printer)
    - [Logging](./src/helper/log.py) -> [or use it directly from the pip packages prime_printer](https://github.com/xXAI-botXx/prime_printer)
    - [Time](./src/helper/time.py) -> [or use it directly from the pip packages prime_printer](https://github.com/xXAI-botXx/prime_printer)
    - [Progress Bar](./src/helper/progress_bar.py) -> [or use it directly from the pip packages prime_printer](https://github.com/xXAI-botXx/prime_printer)


> Note that the license vary. Some code is from me, some from my universities lectures (Prof. Janis Keuper & Prof. Daniela Oelke).





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
    **Condition**:
    ```latex
    x + y ≥ 10
        x ≥ 2
        y ≥ 3
    ```

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
    **Condition**:
    ```latex
    x^2 + y^2 ≥ 10
            x ≥ 2
            y ≥ 3
    ```

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
    **Condition**:
    ```latex
    x + y ≥ 10
        x ≥ 2
        y ≥ 3
    ```

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
    **Condition**:
    ```latex
    x + y ≥ 10
        x ≥ 2
        y ≥ 3
    ```

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
### ONNX

[<img align="right" width=150px src='./res/rackete_2.png'></img>](#ai)

Coming soon...



<br><br>

---
### Experiment Tracking

[<img align="right" width=150px src='./res/rackete_2.png'></img>](#ai)

Coming soon...

(Weights & Biases, ML-Flow)


<br><br>

---
### Visualization

[<img align="right" width=150px src='./res/rackete_2.png'></img>](#ai)

Coming soon...

(Neutron, Zetane, ...)

<br><br>

---
### Using a already made model

[<img align="right" width=150px src='./res/rackete_2.png'></img>](#ai)

In the daily work life of an ai engineer or a ml engineer it is probably the most common way to use an already implemented model. Of course you can program/implement a architecture by yourself and sometimes it is indeed recommended, for example if you want to highly customize the architecture or want to make some parts pretty different. But as stated most likely your starting point is a finish implementation and in most cases you want to change some things, maybe the input, the output, the loss, the dataset or other parts. I made some painful experimence with that and want to share some thoughts here.<br>
1. An implementation most likely have an intended way to use it and I recommend to understand and use this way (which options/parameter are there, how can I add a custom dataset, ...). Building a new way / changing too much of the core pipeline can destroy it and can lead to much suffer and time consumption. So use the implementation as it use to. If you want to change the whole pipeline I recommend to build the implementation from ground up by yourself (yes, I think that will be quicker) but it depends on the situation. So before you try to adjust the code of the implementation to fit to your wished behaviour, it is highly recommended to check if you can adjust your wishes/expectations to let it work, maybe not as you wished but there is a outcome and it works (which is mostlikely the most important thing). For example you want to use a model for predicting a class of a picture and you have a target architecture with implementation and you want to use your model in your pytorch workflow where you have a custom dataset loader, training pipeline with experiment tracking and an evaluation setup, but the implementation does not intend you to use the model directly inside of your code (this is often the case). You can now try to adjust the code to work as you wish and sometimes you don't even have to change something and only have to read and understand the used implementation so that you can use the model, but a quicker and safer way is to try to make a workaround which will be propably also fine. In this case you could use the intended way, calling ht e python script of your implementation (python my_model.py --epochs 50) and you can than use the scripts to let the model generate you the wished output and you build a script to evaluate directly on top of the outputs not the model. Bad thing you maybe have to renounce nice features, as experiment tracking or inside prints/analytics, but it works. <br><br>Real life example: Once I tried to use YOLACT, but I did not wanted to use YOLACT in the intended way (most likely a python script call with arguments) and forked the repository and started to change many things, so that YOLACT could be used and called inside of another python script. I ended up with about 76.100 changed lines ([see the repo](https://github.com/xXAI-botXx/comfortable-yolact)) and the model improved a lot (better prints, usage, ...) but I also somehow destroyed it and the model ended up learning poorly and finding/fixing the problem could take weeks or months...it is difficult to say where the problem comes from, from the new additions, from the deletions, from the old pipeline (whats left from it), the new pipeline, the interactions of them, or on a complete different spot. 
2. Use Anaconda or/and docker. Using stranger code is hard, because every software have many dependencies which are often system specific and maybe outpdated. The available documentation to the installation and usage is often also poorly written and tested, so you need an environment where you can install new python environments quickly and organized. Here I highly recommend using anaconda. With anaconda you can try out different python versions and packages in seconds. Old python versions are also much easier handable than without (trust me, I felt the pain of installing an old python version). Anaconda works also native with the standard pip, so no need to use conda packages. [Check this out for a quick reference](https://github.com/xXAI-botXx/Project-Helper?tab=readme-ov-file#anaconda). Anaconda will make your life much easier in terms of handling dependencies and python versions. Perfect for using another ones implementation.<br>Another recommendation is docker. Docker can start own little virtual machines, with the benefit of less os problems. But of course docker creates a bit of an overhead and makes some things more complicated, so I only recommend docker on more complex or bigger projects or if you have big os issues. [You can check out this reference](https://github.com/xXAI-botXx/Docker).
3. Reach Alternatives. If a model does not want to work and the fixing process takes far too long, you should not hesitate to try other implementations of the same architecture, maybe your own implementation and do not try to hide from another lib tensorflow/pytorch. And if other implementations does not want to work, try another architecture. Just don't waste too much of your valuable time to make one exact implementation working.
4. Don't be scared to adjust implementations for your needs. Yes in the 1. I told you not to do...but hear me out. I would state that it is still the better method to either use an finish implementation in the intended way or build it up on your on in a clean way, but finish implementations still can be changed. The changes should be not too deep and should not change the whole process pipeline, but adding another parameter, adjusting the loss, adding inside prints, logs, experiment tracking, adjust/add your custom dataset, changing input layer, output layer and other things like the learnrate schedular should in most cases not disturb the working software. Just be careful that you don't change core mechanics of the implementation, else you will find yourself fighting against bugs which aquire a understanding of the whole pipeline and in a fixing process over *all* the code.
5. Take your time with an implementation. Read the documentation slowly and look at the scripts and files. You don't have to understand everything, just get a feeling how they implemented it and how they used the model in their own scripts. Then it will be easier to make changes and to use the implementation in your wished way. In this way it is also easy to see the available options/arguments and you see how you cann add new arguments. Using stranger implementations can be hard, often there is not much documentation and the documentation available is poorly and maybe even wrong and outdated. But it is worth it to read the available documentation and fill the gaps with the knowledge of the real code to get a good grid on the implementation.
6. Use available tools. There are plenty websites/tools which allows you to use a model implementation, like Hugging Face, Papers With Code and much other libraries which provide solid and working models. Often the documentation is clear but not deep enough if you want to change deeper things like the output layer. For such changes look at the source code if available. Or use the build-in function help to get your informations `help(MaskRCNN)`. And there are also plenty of tools for other tasks, like deep learning visualization, deployment and experiment tracking.

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












