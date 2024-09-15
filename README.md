# Randomized Least Squares with Subsampled Randomized Hadamard Transform (SRHT)
This project implements randomized least squares regression using Subsampled Randomized Hadamard Transform (SRHT) and compares the results with classical least squares regression. The project includes two test scripts to analyze advertisement datasets: one for TV advertisement budgets and sales, and another for newspaper advertisement budgets and sales.

Here's a brief overview of the mathematical background relevant to your project, focusing on the least squares regression, randomized projections, and the Subsampled Randomized Hadamard Transform (SRHT):

## Mathematical Background

### 1. Least Squares Regression

**Objective:** Minimize the sum of the squared residuals between observed and predicted values.

For a dataset with features $X$ and target $y$, the least squares problem is to find the parameter vector $hat{\beta}$ that minimizes the residual sum of squares (RSS):

$$\hat{\beta} = \arg\min_{\beta} \|X\beta - y\|^2_2$$

where:
- $X$ is an $n \times d$ matrix of features.
- $y$ is an $n \times 1 $ vector of target values.
- $\hat{\beta}$ is the $d \times 1$ vector of coefficients.

The solution is given by:

$$\hat{\beta} = (X^TX)^{-1}X^Ty$$

where $(X^TX)^{-1}X^T$ is known as the Moore-Penrose pseudoinverse of $X$ when $X^TX$ is invertible.

### 2. Randomized Projections

**Objective:** Reduce the dimensionality of the data while approximately preserving the distance between data points.

Randomized projections are used to approximate the original data matrix $X$ by a lower-dimensional matrix $X'$. The projection matrix $R$ is a random matrix that maps the high-dimensional data to a lower-dimensional space:

$$X' = RX$$

where $R$ is a $r \times n$ matrix with $r < n$. 

### 3. Subsampled Randomized Hadamard Transform (SRHT)

**Objective:** Use SRHT to sketch data and apply randomized projections efficiently.

The SRHT is used to project the data matrix $A$ and vector $b$ into a lower-dimensional space while approximating the original problem. The procedure involves:

1. **Hadamard Matrix (H):** A matrix with elements $\pm 1$ that provides an efficient way to perform transformations. It is used to create orthogonal projections.

2. **Diagonal Matrix (D):** A diagonal matrix where the diagonal entries are random variables (Rademacher variables) $\pm 1$.

3. **Sparsified Random Projection Matrix (R):** A matrix used to sample rows from the Hadamard-transformed data. In the uniform version, this matrix is constructed using a uniform distribution; in the non-uniform version, it is constructed using leverage score probabilities.

**Procedure:**

1. Compute the Hadamard matrix $H$ and normalize it.
2. Construct the diagonal matrix $D$ with independent Rademacher random variables.
3. Generate the sparsified random projection matrix $R$. In the non-uniform version, use leverage scores for sampling.
4. Compute the SRHT matrix $RHD$ and apply it to the data matrix $A$ and vector $b$:

   $$A_{\text{sketched}} = RHD \cdot A$$

   $$b_{\text{sketched}} = RHD \cdot b $$

   where $RHD$ is the randomized Hadamard transform matrix.

### 4. Application to Regression

**Objective:** Compare classical least squares regression with SRHT-based randomized regression.

After applying SRHT, the problem is solved in the lower-dimensional space:

1. **Classical Least Squares:** Solve $A\hat{\beta} = b$ using $(A^TA)^{-1}A^Tb$.

2. **Randomized Least Squares with SRHT:** Solve <img src="https://latex.codecogs.com/svg.image?\inline&space;$A_{\text{sketched}}{\hat{\beta}}_{\text{sketched}}=b_{\text{sketched}}$" title="$A_{\text{sketched}}{\hat{\beta}}_{\text{sketched}}=b_{\text{sketched}}$" /> using:


<img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{110}$$\hat{\beta}_{\text{sketched}}=(A_{\text{sketched}}^T&space;A_{\text{sketched}})^{-1}A_{\text{sketched}}^T&space;b_{\text{sketched}}$$" title="$$\hat{\beta}_{\text{sketched}}=(A_{\text{sketched}}^T A_{\text{sketched}})^{-1}A_{\text{sketched}}^T b_{\text{sketched}}$$" />

The quality of the approximation can be measured by comparing the residuals and the L2 difference between the solutions obtained from the classical and randomized methods.
