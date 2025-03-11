# **Logistic Regression for Multiclass Classification (Iris Dataset)**

This project implements **Logistic Regression for Multiclass Classification** using the **One-vs-Rest (OvR) strategy**. The model is trained and evaluated on the **Iris dataset**, which contains three different flower species.

---

## **1. Understanding Logistic Regression**

Logistic Regression is a statistical model used for **classification problems**. Unlike Linear Regression, it applies the **sigmoid function** to map real-valued inputs to probabilities.

### **Sigmoid Function**

The sigmoid function is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

where:
- \(z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b\) (linear combination of weights and features)
- \(w\) represents the model weights, and \(b\) is the bias.
- The output value is in the range **(0,1)**, making it suitable for probability estimation.

---

## **2. Cost Function (Log Loss)**

For a single instance in binary classification, the **log loss** function is:

$$
J(\theta) = - \frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

where:
- \(y_i\) is the actual class label (0 or 1).
- \(\hat{y}_i = \sigma(z)\) is the predicted probability.
- \(n\) is the number of training samples.

For **multiclass classification**, we extend logistic regression using the **One-vs-Rest (OvR) approach**, where we train separate classifiers for each class and pick the one with the highest probability.

---

## **3. Gradient Descent Optimization**

To minimize the cost function, we compute the gradients and update the model parameters iteratively:

### **Gradient Computation**

For each weight \(w_j\) and bias \(b\), the gradients are computed as:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{n} \sum_{i=1}^{n} x_{ij} (\sigma(z_i) - y_i)
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i)
$$

### **Updating Parameters**

Using **gradient descent**, we update the weights and bias:

$$
 w_j = w_j - \alpha \frac{\partial J}{\partial w_j}
$$
$$
 b = b - \alpha \frac{\partial J}{\partial b}
$$

where \(\alpha\) is the **learning rate**.

---

## **4. Implementation Steps**

### **🔹 Dataset: Iris Dataset**
The Iris dataset consists of:
- **150 samples** (50 for each species: Setosa, Versicolor, and Virginica).
- **4 features**: sepal length, sepal width, petal length, petal width.
- **3 output classes**: 0 (Setosa), 1 (Versicolor), 2 (Virginica).

### **🔹 Preprocessing Steps**
1. Load the dataset.
2. Apply **Label Encoding** to convert class names into numeric labels.
3. **Standardize the features** (zero mean, unit variance).
4. Split the dataset into **training and test sets**.

### **🔹 Training the Model**
1. Initialize weights and bias.
2. Train separate **Logistic Regression classifiers** using the **OvR strategy**.
3. Optimize parameters using **gradient descent**.

### **🔹 Making Predictions**
1. Compute the probability for each class using the sigmoid function.
2. Assign each sample to the class with the **highest probability**.

### **🔹 Evaluating Performance**
- **Accuracy Score**: Percentage of correctly classified samples.

---

## **5. Summary**
✅ **Logistic Regression** is used for classification problems by applying the **sigmoid function**.
✅ The model optimizes parameters using **gradient descent** and minimizes **log loss**.
✅ For **multiclass classification**, we use the **One-vs-Rest (OvR) strategy**.
✅ The model is evaluated using the **accuracy metric**.

🚀 **Happy Coding!**

