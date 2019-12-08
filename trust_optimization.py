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

    # Trust Region Problem parameters
    ETA_1 = 0
    ETA_2 = 0.25
    ETA_3 = 0.75
    M_1 = 0.25
    M_2 = 2.0

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
        y_values = y_values.unsqueeze(1) + torch.Tensor(np.random.normal(noise, 0, y_values.shape)).unsqueeze(1)
        x_values = x_values + torch.Tensor(np.random.normal(noise, 0, x_values.shape))
        test_start = train_size + val_size
        return ((x_values[:train_size], y_values[:train_size]),
               (x_values[train_size:train_size + val_size], y_values[train_size:train_size + val_size]),
               (x_values[test_start: test_start + test_size], y_values[test_start: test_start + test_size]))

    def gd_fn_approximation(self, grad, gamma):
        d = self.d(grad, gamma)
        return -(torch.mm(grad, d) + torch.mm(d, d))
    
    def d(self, grad, gamma):
        return -float(1 / float(1 + gamma)) *  grad

    def train_trust(self):
        gamma = 1
        last_val_error = self.run_validation(self.trust_model)
        for epoch in range(self.epochs):
            print("--- Iteration {} ---".format(epoch))
            predictions = self.trust_model(self.train_data[self.DATA_IDX])
            error = self.loss_fn(predictions, self.train_data[self.GT_IDX])

            self.writer.add_scalar("trust_region/train", error.data, epoch)
            error.backward()

            # Temporarily Take a Step
            prev_grad = self.trust_model.weight.grad.clone()
            prev_weights = self.trust_model.weight.clone()
            self.trust_model.weight = torch.nn.Parameter(self.trust_model.weight + self.d(prev_grad, gamma))

            val_error = self.run_validation(self.trust_model)

            self.writer.add_scalar("trust_region/val", val_error, epoch)

            c_numerator = (last_val_error - val_error)
            self.writer.add_scalar("trust_region/val_diff", c_numerator, epoch)

            # Validation Error Increased, and will be stuck
            if c_numerator < 0:
                print("Validation Increased!")


            approx = float(self.gd_fn_approximation(prev_grad, gamma))
            c =  c_numerator / approx
            print("validation_diff: {}  estimation: {} gamma: {} gdfn_approx: {} prev_grad: {}".format(c_numerator, c, gamma, approx, prev_grad))
            if c <= self.ETA_1:  # Step is too large, Abort.
                print("c too small, aborting...")
                self.trust_model.weight = torch.nn.Parameter(prev_weights)
            else:
                last_val_error = val_error
                

            if c < self.ETA_2:
                gamma /= self.M_1
            elif self.ETA_3 < c:
                gamma /= self.M_2 
            self.writer.add_scalar("trust_region/gamma", gamma, epoch)

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