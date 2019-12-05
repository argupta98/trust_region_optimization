import torch
import numpy as np

def generate_data(dim=1, degree=1, noise=0):
    pass

def train_trust(model, data):
    pass

def train_normal(model, data):
    pass

def make_linear_regression(input_dim=1, output_dim=1):
    return torch.nn.Linear(input_dim, output_dim)

def make_mlp():
    pass

def copy_model(model):
    pass

def evaluate(model):
    pass

if __name__ == "__main__":
    trust_model = make_linear_regression() 
    model = copy_model(trust_model)
    data = generate_data()
    train_trust(trust_model)
    train_normal(model)
    evaluate(trust_model)
    evaluate(model)