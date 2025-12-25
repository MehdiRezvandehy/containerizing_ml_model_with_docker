# Containerizing ML Model with Docker

This repository demonstrates how to **containerize, and deploy a machine learning model** using Docker. It focuses on best practices for packaging ML applications in a reproducible and portable way.

## 🚀 Project Summary

This project includes:

* Training an **XGBoost model** and save the trained model as `.pkl` files
* Containerizing the application using:

  * **Manual containerization** (learning exercise)
  * **Dockerfile-based containerization** (recommended)
* Building and pushing a Docker image to **Docker Hub**
* Deploying the containerized app to **Hugging Face Spaces (Free Tier)**

## 🧠 What You’ll Learn

* How Docker containers work
* How to write a Dockerfile for an ML app
* How to build, tag, and push Docker images
* How to deploy a container on a hosted platform

## 📦 How It Works

### 1. Train & Save the Model

* Code loads and preprocesses training data
* Trains an XGBoost model
* Serializes and saves the model as `model.pkl`

This file can be loaded later for inference inside a container.

### 2. Manual Containerization

* Run the app inside a container
* Once configured, commit the container to create an image: 
```bash
docker container commit
```
* *This method is educational and not recommended for production.*

### 3. Dockerfile Containerization

* Write a `Dockerfile` defining the base image, dependencies, and startup command
* Build a reproducible Docker image:

```bash
docker build -t your_image_name .
```

### 4. Publish to Docker Hub

* Log in to Docker Hub
* Tag and push the image to your Docker Hub repo:

```bash
docker tag your_image_name username/your_image_name
docker push username/your_image_name
```

### 5. Deploy on Hugging Face Spaces

* Create a new Space with Docker runtime
* Configure it to pull your Docker image
* Deploy and test the hosted app


If you paste the **actual directory structure and filenames**, I can tailor this further with exact commands and paths from your repo.
