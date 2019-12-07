import torch
import numpy as np
import copy
from tensorboardX import SummaryWriter

class TrustRegionTester(object):
    TRAIN_PCT = 0.5
    VAL_PCT = 0.2
    TEST_PCT = 0.3
    DATA_IDX = 0
    GT_IDX = 1

    def __init__(self):
        # Setup models
        self.trust_model = self.make_linear_regression() 
        self.model = self.make_linear_regression()
        self.joint_model = self.make_linear_regression()

        # Copy over intial weights, so we know they start at the same state.
        self.model.load_state_dict(copy.deepcopy(self.trust_model.state_dict()))
        self.joint_model.load_state_dict(copy.deepcopy(self.trust_model.state_dict()))
        self.loss_fn = torch.nn.MSELoss()

        self.writer = SummaryWriter()

        # Generate Data
        self.train_data, self.val_data, self.test_data = self.generate_data()

        self.epochs = 100
        self.lr = 0.0001

    def train(self):
        self.train_trust()
        self.train_normal()
    
    def evaluate(self):
        with torch.no_grad():
            for name, model in [("trust", self.trust_model), ("normal", self.model)]:
                model.eval()
                test_predictions = model(self.test_data[self.DATA_IDX])
                loss = self.loss_fn(test_predictions, self.test_data[self.GT_IDX])
                print("evaluation for {}: {}".format(name, loss.data))

    def generate_data(self, dim=1, degree=1, noise=0, num_samples=500):
        """For now just generate data in a single line with no noise.""" 
        # Line is y = 4x + 6 
        # Sample x's uniformly at random from [-100, 100] 
        train_size = int(num_samples * self.TRAIN_PCT)
        val_size = int(num_samples * self.VAL_PCT)
        test_size = int(num_samples * self.TEST_PCT)
        x_values = torch.Tensor(np.random.randint(-100, 100, num_samples))
        y_values =  4 * x_values + 6
        x_values = x_values.unsqueeze(1)
        y_values = y_values.unsqueeze(1)
        test_start = train_size + val_size
        return ((x_values[:train_size], y_values[:train_size]),
               (x_values[train_size:train_size + val_size], y_values[train_size:train_size + val_size]),
               (x_values[test_start: test_start + test_size], y_values[test_start: test_start + test_size]))

    def gd_fn_approximation(self, grad, gamma):
        return (2 + gamma) * torch.mm(grad, grad)

    def train_trust(self):
        gamma = 1
        last_val_error = self.run_validation(self.trust_model)
        for epoch in range(self.epochs):
            predictions = self.trust_model(self.train_data[self.DATA_IDX])
            error = self.loss_fn(predictions, self.train_data[self.GT_IDX])

            self.writer.add_scalar("trust_region/train", error.data, epoch)
            error.backward()

            # Temporarily Take a Step
            prev_grad = torch.Tensor(self.trust_model.weight.grad)
            self.trust_model.weight = torch.nn.Parameter(self.trust_model.weight - self.lr * (1 / float(1 + gamma)) * prev_grad)

            val_error = self.run_validation(self.trust_model)

            self.writer.add_scalar("trust_region/val", val_error, epoch)

            d_diff_err = 1
            c_numerator = (last_val_error - val_error)

            # Validation Error Increased, and will be stuck
            if c_numerator < 0:
                print("Validation Increased!")

            c =  c_numerator / float(self.gd_fn_approximation(prev_grad, gamma))
            print("estimation: {} gamma: {}".format(c, gamma))
            if c < 1:  # Step is too large, Abort.
                gamma /= 2.0
                self.trust_model.weight = torch.nn.Parameter(self.trust_model.weight + self.lr * (1 / float(1 + gamma)) * prev_grad)

    def train_normal(self):
        top_validation = 10**10
        top_model = None
        for epoch in range(self.epochs):
            predictions = self.model(self.train_data[self.DATA_IDX])
            error = self.loss_fn(predictions, self.train_data[self.GT_IDX])
            self.writer.add_scalar("normal/train", error.data, epoch)
            error.backward()
            self.model.weight = torch.nn.Parameter(self.model.weight - self.lr * self.model.weight.grad)
            val_error = self.run_validation(self.model)
            self.writer.add_scalar("normal/val", val_error, epoch)
            if val_error < top_validation:
                top_validation = val_error
                top_model = self.model
        self.model = top_model
        print("top normal model validation: {}".format(top_validation))
    
    def train_joint(self):
        pass

    def run_validation(self, model):
        with torch.no_grad():
            predictions = model(self.val_data[self.DATA_IDX])
            error = self.loss_fn(predictions, self.val_data[self.GT_IDX])
            return error.data

    def make_linear_regression(self, input_dim=1, output_dim=1):
        return torch.nn.Linear(input_dim, output_dim)

    def make_mlp():
        pass


if __name__ == "__main__":
    trust_tester = TrustRegionTester()
    trust_tester.train()
    trust_tester.evaluate()