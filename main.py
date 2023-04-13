from src.training import training
from src.prediction import prediction
from flask import Flask, request, render_template


if __name__ == "__main__":
    training("params.yaml")
    prediction("params.yaml")

