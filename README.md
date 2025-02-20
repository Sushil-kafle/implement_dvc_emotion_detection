# MLOps for Emotion Detection

This project focuses on Emotion Detection and uses **DVC** pipelines, data versioning, and **Git** to streamline the machine learning workflow. By utilizing **AWS S3** for remote storage, it efficiently manages datasets and models, enabling smooth collaboration and deployment.

# 📌 Setup Guide

Follow these steps to set up your environment and manage your data with DVC:

## 🚀 Installation & Setup

### 1️⃣ **Install `uv` on your system**

```sh
    pip install uv
```

### 2️⃣ **Sync dependencies**

```sh
  uv sync
```

### 3️⃣ **Configure environment variables**

- Create a .env file based on .example.env

## 🌐 Setup Kaggle API for Data Download

### 4️⃣ **Set up Kaggle API key**

- Go to your Kaggle account settings
- Create New API Token
- update the environment variable

## ☁️ Configure DVC with S3 Remote Storage

### 5️⃣ **Set up an S3 bucket for DVC storage**

- Create an S3 bucket on AWS or use an existing one.
- Update your .env file with the S3 bucket details.

### 6️⃣ **Configure DVC to use the S3 bucket for remote storage**

```sh
  dvc remote add -d s3remote s3://personalbucket111

```

## 📦 Data Management with DVC

### 7️⃣ **Fetch the latest DVC-tracked data**

```sh
  dvc fetch
```

### 8️⃣ **Checkout the data files**

```sh
  dvc checkout
```

### 9️⃣ **Run pipeline again**

```sh
  dvc repro
```
