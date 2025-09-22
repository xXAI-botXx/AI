# Math for AI/Deep Learning

Content:
- [Linear Algebra](#linear-algebra)
- [Probability](#probability)

<br>

> May still be advanced with details. Currently added only a little summary of each topic.

<br>

---
## Linear Algebra

### Scalars, Vectors, Matrixes and Tensors

Scalars, vectors, matrices, and tensors are the fundamental building blocks of linear algebra. They are ways of storing and organizing data.

* **Scalars**  
    A scalar is a single number. Example: $s = 5$ or $s = 3.14$.  
    Scalars are often real numbers $s \in \mathbb{R}$, but depending on context, they can also be integers $\mathbb{Z}$ or natural numbers $\mathbb{N}$.
* **Vectors**  
    A vector is an ordered list of numbers (1-D array).  
    Example: $\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} \in \mathbb{R}^3$  
    Vectors represent points, directions, or features in a space.
* **Matrices**  
    A matrix is a 2-D array of numbers with rows and columns.  
    Example:  
    $$
    \mathbf{M} =
    \begin{bmatrix}
    m_{11} & m_{12} \\
    m_{21} & m_{22} \\
    m_{31} & m_{32}
    \end{bmatrix}
    \in \mathbb{R}^{3 \times 2}
    $$
    Matrices are used to represent linear transformations or datasets.
* **Tensors**  
    A tensor generalizes vectors and matrices to higher dimensions (n-D arrays).  
    - 0th order tensor: scalar  
    - 1st order tensor: vector  
    - 2nd order tensor: matrix  
    - 3rd order tensor (or higher): multi-dimensional arrays

    Tensors are heavily used in deep learning to represent batches of data, e.g., images as 4D tensors: (batch, height, width, channels).


<br><br>

---
### Multiplication of Matrices and Vectors

Matrix–vector multiplication applies a linear transformation to a vector.  

If $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{x} \in \mathbb{R}^n$, then  

$$
\mathbf{y} = \mathbf{A} \mathbf{x} \quad \Rightarrow \quad \mathbf{y} \in \mathbb{R}^m
$$  

Example:  
$$
\begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6
\end{bmatrix}
\begin{bmatrix}
7 \\
8
\end{bmatrix}
=
\begin{bmatrix}
23 \\
53 \\
83
\end{bmatrix}
$$

<br><br>

---
### Identity and Inverse Matrices

* **Identity Matrix**  
    Acts like the number `1` for matrices.  
    $$
    \mathbf{I}_n =
    \begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
    \end{bmatrix}
    \quad
    \mathbf{I}_n \mathbf{x} = \mathbf{x}
    $$

* **Inverse Matrix**  
    For a square matrix $\mathbf{A}$, the inverse $\mathbf{A}^{-1}$ satisfies:  
    $$
    \mathbf{A}\mathbf{A}^{-1} = \mathbf{I}
    $$  
    Not all matrices are invertible (those with $\det(\mathbf{A}) = 0$ are not).

<br><br>

---
### Linear Algebra and Linear Envelope

A **linear combination** of vectors is a weighted sum:  
$$
\alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \dots + \alpha_k \mathbf{v}_k
$$

The set of all possible linear combinations is the **span** (or envelope).  
This defines the subspace that the vectors can cover.

<br><br>

---
### Length of Vectors (Norms)

The **norm** measures the size (length) of a vector.  

* **L2 norm (Euclidean length):**  
    $$
    \|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2}
    $$

* **L1 norm (Manhattan length):**  
    $$
    \|\mathbf{v}\|_1 = |v_1| + |v_2| + \dots + |v_n|
    $$

Norms are crucial for measuring error, regularization, and distances.

<br><br>

---
### Special Matrices and Vectors

* **Diagonal matrix:** Non-zero entries only on the diagonal.  
* **Symmetric matrix:** $\mathbf{A} = \mathbf{A}^\top$.  
* **Orthogonal matrix:** $\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$.  
* **One-hot vector:** All zeros except one entry equal to $1$.  
* **Basis vectors:** Unit vectors aligned with coordinate axes.

<br><br>

---
### Eigenvalue Decomposition

For a square matrix $\mathbf{A}$:  
$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}
$$  

- $\lambda$ = eigenvalue  
- $\mathbf{v}$ = eigenvector  

This shows directions ($\mathbf{v}$) that are only scaled, not rotated, by $\mathbf{A}$.

<br><br>

---
### Singular Value Decomposition (SVD)

Any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ can be factorized as:  
$$
\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top
$$

- $\mathbf{U}$ and $\mathbf{V}$ are orthogonal matrices.  
- $\mathbf{\Sigma}$ is diagonal with singular values.  

SVD is used for dimensionality reduction (e.g., PCA) and matrix compression.

<br><br>

---
### Moore–Penrose Pseudoinverse

For non-square or non-invertible matrices, the pseudoinverse generalizes the inverse:  
$$
\mathbf{A}^+ = \mathbf{V} \mathbf{\Sigma}^+ \mathbf{U}^\top
$$

It is often used in least squares problems.

<br><br>

---
### Trace Operator

The **trace** of a square matrix is the sum of its diagonal elements:  
$$
\text{Tr}(\mathbf{A}) = \sum_i a_{ii}
$$

It is invariant under cyclic permutations:  
$$
\text{Tr}(\mathbf{AB}) = \text{Tr}(\mathbf{BA})
$$

<br><br>

---
### Determinant

The **determinant** of a square matrix measures how much it scales volume:  
$$
\det(\mathbf{A})
$$  

- $\det(\mathbf{A}) = 0 \implies$ matrix not invertible.  
- In 2D: absolute value of the parallelogram area spanned by the rows (or columns).  
- In 3D: absolute value of the parallelepiped volume.


<br><br>

---
## Probability

### Why Probability? (in Deep Learning)

Deep learning often deals with **uncertainty**:  
- Data can be noisy or incomplete.  
- Models need to make predictions (not always exact).  
- Training involves randomness (e.g., initialization, dropout, stochastic optimization).  

Probability provides the **language of uncertainty** that helps us design and analyze models.  
Examples:  
- Softmax output = probability distribution over classes.  
- Loss functions like **cross-entropy** are derived from probability theory.  
- Bayesian deep learning explicitly models uncertainty.

<br><br>

---
### Random Variables

A **random variable (RV)** assigns a number to an outcome of a random experiment.  

- **Discrete RV**: finite or countable outcomes (e.g., dice roll).  
    Example: $X \in \{1,2,3,4,5,6\}$.  
- **Continuous RV**: infinitely many outcomes (e.g., real numbers from sensor noise).  
    Example: $Y \in \mathbb{R}$.

Notation:  
- Discrete: $P(X = x)$  
- Continuous: probability density function (PDF) $p(x)$, with  
    $$
    P(a \leq X \leq b) = \int_a^b p(x) \, dx
    $$

<br><br>

---
### Probability Distributions

A **distribution** describes how likely different outcomes are.

**Discrete Variables and Probability Mass Functions (PMFs):**  
- $P(X = x)$ gives probability of outcome $x$.  
- Example: fair die: $P(X=i) = \frac{1}{6}, \; i=1,...,6$.

**Continuous Variables and Probability Density Functions (PDFs):**  
- Probability of an exact value is $0$.  
- Probabilities come from integrals of the density.  
- Example: Gaussian (Normal distribution).

<br><br>

---
### Marginal Probability

The probability of a subset of variables, ignoring the others.  

If $P(X, Y)$ is a joint distribution:  
$$
P(X=x) = \sum_y P(X=x, Y=y) \quad \text{(discrete)}
$$  
$$
p(x) = \int p(x,y) \, dy \quad \text{(continuous)}
$$  

Marginalization = “averaging out” irrelevant variables.

<br><br>

---
### Conditional Probabilities

Probability of an event given another event has occurred:  
$$
P(A|B) = \frac{P(A, B)}{P(B)}
$$

Example: Probability of rain given cloudy skies.

<br><br>

---
### Product Rule in Conditional Probabilities

The **product rule** relates joint and conditional probability:  
$$
P(A, B) = P(A|B) P(B) = P(B|A) P(A)
$$  

This is the foundation of Bayesian reasoning.

<br><br>

---
### Independence and Conditional Independence

* **Independence:**  
    $P(A, B) = P(A) P(B)$  
    Events don’t affect each other.  
* **Conditional Independence:**  
    $P(A, B | C) = P(A|C) P(B|C)$  
    Once $C$ is known, $A$ and $B$ are independent.  

Example in ML: Given the class label, two features may be conditionally independent.

<br><br>

---
### Expected Value, Variance, Covariance

* **Expectation (mean):**  
    Discrete: $\mathbb{E}[X] = \sum_x x P(X=x)$  
    Continuous: $\mathbb{E}[X] = \int x p(x) dx$
* **Variance (spread of values):**  
    $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$
* **Covariance (relation between variables):**  
    $\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$

Covariance matrices are crucial in PCA and Gaussian distributions.

<br><br>

---
### Popular Probability Distributions

* **Bernoulli Distribution:**  
    One trial, success/failure. $P(X=1) = p, \; P(X=0)=1-p$.
* **Multinoulli (Categorical):**  
    Multi-class outcomes, used in softmax layers.
* **Normal (Gaussian) Distribution:**  
    Most important continuous distribution.  
    $$
    p(x) = \frac{1}{\sqrt{2\pi \sigma^2}} e^{-\frac{(x-\mu)^2}    {2\sigma^2}}
    $$  
    Used for modeling noise, priors, and weights initialization.
* **Exponential and Laplace Distributions:**  
    Model time between events, or heavy-tailed distributions.
* **Dirac-Delta and Empirical Distributions:**  
    - Dirac-Delta: idealized “point mass” distribution.  
    - Empirical: distribution built from observed samples.
    * **Combined Distributions:**  
    Complex models built by combining simpler ones (mixture models, Gaussian Mixture Models).

<br><br>

---
### Popular Functions in Deep Learning

Many functions in deep learning have probabilistic interpretations:  
- **Sigmoid:** maps real values to $[0,1]$, interpretable as probability.  
- **Softmax:** converts logits into a categorical distribution.  
- **Cross-Entropy Loss:** measures distance between predicted and true distributions.  
- **KL-Divergence:** compares probability distributions.

<br><br>

---
### Bayes’ Theorem

A fundamental rule for updating beliefs:  
$$
P(A|B) = \frac{P(B|A) P(A)}{P(B)}
$$

Used in Bayesian deep learning, probabilistic models, and evidence-based reasoning.

<br><br>

---
### Technical Details of Continuous Variables

- Continuous variables are described by **densities** $p(x)$, not probabilities.  
- Integrals replace sums.  
- Change of variables requires **Jacobian determinant**.  

Important in **normalizing flows** and **variational inference**.

<br><br>

---
### Information Theory

* **Entropy:** measure of uncertainty.  
    $$
    H(X) = - \sum_x P(x) \log P(x)
    $$
* **Cross-Entropy:** used as a loss function in classification.  
* **KL Divergence:** asymmetrical distance between distributions.  

Information theory explains why certain loss functions are used in deep learning.

<br><br>

---
### Structured Probabilistic Models

Probabilistic graphical models capture dependencies among variables.  

- **Bayesian Networks:** directed acyclic graphs with conditional probabilities.  
- **Markov Random Fields:** undirected graphs capturing spatial relationships.  
- **Hidden Markov Models (HMMs):** sequential data with hidden states.  

In deep learning, structured models inspire architectures like RNNs and attention mechanisms.

<br><br>

---






