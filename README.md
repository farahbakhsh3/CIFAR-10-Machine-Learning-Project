# CIFAR-10 Machine Learning Project

This repository contains a compact end-to-end study of the CIFAR-10 image classification dataset using multiple modeling approaches. The project compares a simple convolutional neural network, a residual CNN, and several classical machine learning baselines, and it includes evaluation utilities for reporting accuracy, confusion matrices, and per-class performance.

## Project Overview

The CIFAR-10 dataset consists of 60,000 color images of size $32 \times 32$ pixels divided into 10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The notebooks in this repository train and evaluate models for this task and save the best-performing checkpoints as HDF5 files.

## What Is Included

- A simple CNN implementation in [cnn_simple.ipynb](cnn_simple.ipynb)
- A residual CNN implementation in [cnn_resnet.ipynb](cnn_resnet.ipynb)
- A classical machine learning benchmark in [classical_ml.ipynb](classical_ml.ipynb)
- Evaluation helpers in [model_check.py](model_check.py)
- Pretrained model checkpoints:
  - [best_cifar10_cnn.h5](best_cifar10_cnn.h5)
  - [best_cifar10_resnet.h5](best_cifar10_resnet.h5)

## Repository Structure

```text
.
├── README.md
├── model_check.py
├── cnn_simple.ipynb
├── cnn_resnet.ipynb
├── classical_ml.ipynb
├── best_cifar10_cnn.h5
├── best_cifar10_resnet.h5
└── LICENSE
```

## Dataset

The notebooks use the CIFAR-10 dataset provided by Keras through `tensorflow.keras.datasets.cifar10`.

Typical preprocessing steps include:

- loading train and test splits
- normalizing pixel values to the range $[0, 1]$
- converting labels to one-hot vectors for neural network training
- resizing or reshaping inputs when required by specific models

## Models and Experiments

### 1. Simple CNN

The simple CNN notebook trains a compact convolutional neural network with stacked convolutional layers, pooling layers, and dense layers. This serves as a baseline deep learning model for CIFAR-10 classification.

### 2. Residual CNN

The residual CNN notebook uses a deeper architecture with residual blocks to improve feature learning and gradient flow. This model is more expressive than the simple CNN and generally achieves stronger performance.

### 3. Classical Machine Learning

The classical ML notebook evaluates several non-deep-learning classifiers such as:

- Logistic Regression
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine
- Multi-Layer Perceptron
- XGBoost

These experiments provide a traditional machine learning benchmark for comparison with the neural network approaches.

## Evaluation Utilities

The script [model_check.py](model_check.py) provides helper functions for:

- generating full classification reports
- printing confusion matrices
- computing per-class accuracy
- visualizing predictions and misclassifications

These utilities are especially useful after loading a trained model to inspect its behavior on the test set.

## Installation

This project is designed for Python 3.9+.

Install the required packages with:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter ipykernel
```

If you are using a GPU-enabled environment, TensorFlow will leverage it automatically when available.

## Running the Notebooks

Start Jupyter Notebook or JupyterLab from the repository root:

```bash
jupyter notebook
```

Then open one of the notebooks:

- [cnn_simple.ipynb](cnn_simple.ipynb)
- [cnn_resnet.ipynb](cnn_resnet.ipynb)
- [classical_ml.ipynb](classical_ml.ipynb)

## Reproducing the Training Workflow

1. Open the desired notebook.
2. Run the cells in order.
3. The notebook will download the CIFAR-10 dataset, train the model, and save the best checkpoint.
4. The final cells evaluate the saved model on the test split and print performance metrics.

## Loading a Saved Model

You can load the saved checkpoints with TensorFlow:

```python
import tensorflow as tf

model = tf.keras.models.load_model("best_cifar10_cnn.h5")
print(model.summary())
```

## Example Results

The notebooks report strong performance for the deep learning models, while the classical methods provide a useful baseline. Representative results observed in the training notebooks include:

- Simple CNN: approximately 89.67% test accuracy
- Residual CNN: approximately 91.77% test accuracy
- Classical ML baselines: accuracy varied by model, with SVM and MLP performing better than simpler methods

These values may vary slightly depending on hardware, random initialization, and training conditions.

## Notes

- The repository is intentionally lightweight and focused on experimentation and education.
- The project is well-suited for learning image classification, CNN architectures, transfer of concepts from classical machine learning to deep learning, and model evaluation.
- The saved model files can be used for quick inference or comparison without retraining.

## License

This project is distributed under the terms of the license included in [LICENSE](LICENSE).

--------------------------------

# پروژه یادگیری ماشین CIFAR-10

این مخزن، مطالعه‌ای جامع و پایان‌به‌پایان (End-to-End) بر روی مجموعه داده طبقه‌بندی تصویر CIFAR-10 را شامل می‌شود. پروژه رویکردهای مختلف مدل‌سازی را مقایسه می‌کند: یک شبکه عصبی کانولوشنال (CNN) ساده، یک CNN باقی‌مانده (Residual CNN) و چندین مدل پایه یادگیری ماشین کلاسیک. این پروژه همچنین شامل ابزارهای ارزیابی برای گزارش دقت (Accuracy)، ماتریس درهم‌ریختگی (Confusion Matrix) و عملکرد به ازای هر کلاس است.

## نمای کلی پروژه

مجموعه داده CIFAR-10 شامل ۶۰,۰۰۰ تصویر رنگی با ابعاد $32 	imes 32$ پیکسل است که به ۱۰ کلاس تقسیم شده‌اند:

- هواپیما (airplane)
- خودرو (automobile)
- پرنده (bird)
- گربه (cat)
- گوزن (deer)
- سگ (dog)
- قورباغه (frog)
- اسب (horse)
- کشتی (ship)
- کامیون (truck)

نوت‌بوک‌های این مخزن، مدل‌هایی را برای این وظیفه آموزش داده و ارزیابی می‌کنند و بهترین مدل‌های آموزش‌دیده (checkpoints) را در قالب فایل‌های HDF5 ذخیره می‌کنند.

## محتویات

- پیاده‌سازی یک CNN ساده در [cnn_simple.ipynb](cnn_simple.ipynb)
- پیاده‌سازی یک CNN باقی‌مانده در [cnn_resnet.ipynb](cnn_resnet.ipynb)
- یک بنچمارک یادگیری ماشین کلاسیک در [classical_ml.ipynb](classical_ml.ipynb)
- ابزارهای کمکی ارزیابی مدل در [model_check.py](model_check.py)
- چک‌پوینت‌های مدل از پیش آموزش‌دیده:
  - [best_cifar10_cnn.h5](best_cifar10_cnn.h5)
  - [best_cifar10_resnet.h5](best_cifar10_resnet.h5)

## ساختار مخزن

```text
.
├── README.md
├── README_FA.md
├── model_check.py
├── cnn_simple.ipynb
├── cnn_resnet.ipynb
├── classical_ml.ipynb
├── best_cifar10_cnn.h5
├── best_cifar10_resnet.h5
└── LICENSE
```

## مجموعه داده (Dataset)

نوت‌بوک‌ها از مجموعه داده CIFAR-10 استفاده می‌کنند که توسط Keras از طریق `tensorflow.keras.datasets.cifar10` ارائه می‌شود.

مراحل پیش‌پردازش معمول شامل موارد زیر است:

- بارگذاری داده‌های آموزش و تست.
- نرمال‌سازی مقادیر پیکسل در بازه $[0, 1]$.
- تبدیل برچسب‌ها به بردارهای One-Hot برای آموزش شبکه‌های عصبی.
- تغییر اندازه یا تغییر شکل ورودی‌ها در صورت نیاز مدل‌های خاص.

## مدل‌ها و آزمایش‌ها

### ۱. CNN ساده
نوت‌بوک CNN ساده، یک شبکه عصبی کانولوشنال فشرده با لایه‌های کانولوشنال، پولینگ و متراکم را آموزش می‌دهد. این به عنوان یک مدل پایه یادگیری عمیق برای طبقه‌بندی CIFAR-10 عمل می‌کند.

### ۲. CNN باقی‌مانده (Residual CNN)
نوت‌بوک Residual CNN از معماری عمیق‌تری با بلوک‌های باقی‌مانده برای بهبود یادگیری ویژگی و جریان گرادیان استفاده می‌کند. این مدل نسبت به CNN ساده بیانگرتر است و معمولاً عملکرد قوی‌تری ارائه می‌دهد.

### ۳. یادگیری ماشین کلاسیک
نوت‌بوک ML کلاسیک، طبقه‌بندی‌کننده‌های مختلفی مانند موارد زیر را ارزیابی می‌کند:
- رگرسیون لجستیک (Logistic Regression)
- K-نزدیک‌ترین همسایه (K-Nearest Neighbors)
- بیز ساده (Naive Bayes)
- درخت تصمیم (Decision Tree)
- جنگل تصادفی (Random Forest)
- ماشین بردار پشتیبان (Support Vector Machine)
- پرسپترون چندلایه (Multi-Layer Perceptron)
- XGBoost

این آزمایش‌ها یک بنچمارک یادگیری ماشین سنتی را برای مقایسه با رویکردهای شبکه عصبی فراهم می‌کنند.

## ابزارهای ارزیابی
اسکریپت [model_check.py](model_check.py) توابع کمکی برای موارد زیر را فراهم می‌کند:
- تولید گزارش کامل طبقه‌بندی.
- چاپ ماتریس درهم‌ریختگی.
- محاسبه دقت به ازای هر کلاس.
- بصری‌سازی پیش‌بینی‌ها و طبقه‌بندی‌های نادرست.
این ابزارها به ویژه پس از بارگذاری یک مدل آموزش‌دیده برای بازرسی رفتار آن بر روی مجموعه تست مفید هستند.

## نصب و راه‌اندازی

این پروژه برای Python 3.9+ طراحی شده است.

بسته‌های مورد نیاز را با دستور زیر نصب کنید:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter ipykernel
```

اگر از محیطی با قابلیت GPU استفاده می‌کنید، TensorFlow به طور خودکار از آن بهره خواهد برد.

### اجرای نوت‌بوک‌ها
Jupyter Notebook یا JupyterLab را از ریشه مخزن اجرا کنید:

```bash
jupyter notebook
```

سپس یکی از نوت‌بوک‌ها را باز کنید:
- [cnn_simple.ipynb](cnn_simple.ipynb)
- [cnn_resnet.ipynb](cnn_resnet.ipynb)
- [classical_ml.ipynb](classical_ml.ipynb)

## بازتولید گردش کار آموزش

۱. نوت‌بوک مورد نظر را باز کنید.
۲. سلول‌ها را به ترتیب اجرا کنید.
۳. نوت‌بوک مجموعه داده CIFAR-10 را دانلود، مدل را آموزش داده و بهترین چک‌پوینت را ذخیره خواهد کرد.
۴. سلول‌های پایانی، مدل ذخیره شده را بر روی مجموعه تست ارزیابی کرده و معیارهای عملکرد را چاپ می‌کنند.

## بارگذاری مدل ذخیره شده
می‌توانید چک‌پوینت‌های ذخیره شده را با TensorFlow بارگذاری کنید:

```python
import tensorflow as tf

model = tf.keras.models.load_model("best_cifar10_cnn.h5")
print(model.summary())
```

## نتایج نمونه
نوت‌بوک‌ها عملکرد قوی برای مدل‌های یادگیری عمیق گزارش می‌دهند، در حالی که روش‌های کلاسیک یک پایه مفید را ارائه می‌دهند. نتایج نماینده مشاهده شده در نوت‌بوک‌های آموزشی عبارتند از:

- **CNN ساده:** دقت تست تقریباً ۸۹.۶۷٪
- **Residual CNN:** دقت تست تقریباً ۹۱.۷۷٪
- **پایه‌های ML کلاسیک:** دقت بسته به مدل متفاوت بود، به طوری که SVM و MLP عملکرد بهتری نسبت به روش‌های ساده‌تر داشتند.

این مقادیر ممکن است بسته به سخت‌افزار، مقداردهی اولیه تصادفی و شرایط آموزشی کمی متفاوت باشند.

## نکات
- این مخزن عمداً سبک و متمرکز بر آزمایش و آموزش طراحی شده است.
- این پروژه برای یادگیری طبقه‌بندی تصویر، معماری‌های CNN، انتقال مفاهیم از یادگیری ماشین کلاسیک به یادگیری عمیق و ارزیابی مدل بسیار مناسب است.
- فایل‌های مدل ذخیره شده را می‌توان برای استنتاج سریع یا مقایسه بدون نیاز به بازآموزی استفاده کرد.

## مجوز (License)
این پروژه تحت شرایط مجوز موجود در [LICENSE](LICENSE) توزیع شده است.
