# Deep Learning: Lessons, Errors, and Mathematical Insights

## 1. Understanding Deep Learning Errors

### 1.1 Loss Functions and Their Behavior

Loss functions measure how well the neural network performs. Common choices include:

**Mean Squared Error (MSE):**
- Used for regression; penalizes large errors heavily
- Formula: `MSE = (1/n) * Σ(yi - ŷi)²`

**Cross-Entropy Loss:**
- Used for classification; derived from information theory
- Formula: `Cross-Entropy = -Σ(yi * log(ŷi))`

**Huber Loss:**
- Robust to outliers by combining MSE and MAE
- Formula: `Huber(x) = (1/2)x² for |x| ≤ δ, δ|x| - (1/2)δ² otherwise`

### 1.2 Overfitting vs. Underfitting

**Overfitting:**
- High training accuracy, poor test accuracy
- Signs: Model performs well on training data but fails to generalize
- Solutions: Regularization, dropout, early stopping

**Underfitting:**
- Poor performance on both training and test data
- Cause: Insufficient model capacity
- Solutions: Increase model complexity, add features

**Regularization Techniques:**
- L1 Regularization: `Lreg = L + λ|W|`
- L2 Regularization: `Lreg = L + λ||W||²`
- Dropout: Randomly deactivate neurons during training

### 1.3 Vanishing & Exploding Gradients

**Common Issues:**
- Vanishing: Gradients become too small (common with sigmoid/tanh)
- Exploding: Gradients become too large (poor initialization)

**Solutions:**
- Use ReLU activation: `f(x) = max(0, x)`
- Implement batch normalization
- Apply proper weight initialization techniques

## 2. Data Preprocessing & Challenges

### 2.1 Handling Missing Data

**Techniques:**
- Mean/median imputation: Replace missing values with average
- Deletion: Remove rows with missing data
- Interpolation: Estimate using neighboring values

### 2.2 Normalization vs. Standardization

**Normalization:**
- Scales features to range [0,1]
- Formula: `x' = (x - min(x))/(max(x) - min(x))`

**Standardization:**
- Scales to zero mean and unit variance
- Formula: `x' = (x - μ)/σ`

## 3. Optimization & Training Stability

### 3.1 Learning Rate Optimization

**Optimizers:**
- SGD: Basic gradient descent with momentum
- Adam: Adaptive moment estimation
- RMSprop: Root mean square propagation

**Learning Rate Schedules:**
- Linear decay: `lr = initial_lr * (1 - epoch/total_epochs)`
- Cosine decay: `lr = initial_lr * (1 + cos(epoch * π/total_epochs))/2`
- Step decay: Reduce by factor every n epochs

### 3.2 Advanced Training Techniques

**Batch Normalization:**
- Normalizes layer inputs
- Reduces internal covariate shift
- Formula: `BN(x) = γ * (x - μ)/σ + β`

**Dropout:**
- Rate typically between 0.2-0.5
- Only active during training
- Scales activations by 1/(1-p) during inference

## 4. Debugging Strategies

### 4.1 Common Issues & Solutions

**Loss Explosion:**
- Check learning rate magnitude
- Verify input data normalization
- Inspect gradient magnitudes

**Plateauing Loss:**
- Adjust learning rate schedule
- Implement learning rate warmup
- Consider architecture changes

### 4.2 Monitoring Tools

**Essential Metrics:**
- Training/validation loss curves
- Layer activation distributions
- Gradient magnitudes
- Model parameter statistics

**Visualization:**
- TensorBoard for real-time monitoring
- Confusion matrices for classification
- Feature importance plots
- Learning rate vs. loss curves
