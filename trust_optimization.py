import torch
import numpy as np
from tensorboardX import SummaryWriter

class TrustRegionTester(object):
    TRAIN_PCT = 0.5
    VAL_PCT = 0.2
    TEST_PCT = 0.3
    DATA_IDX = 0
    GT_IDX = 1

    def __init__(self):
        self.trust_model = make_linear_regression() 
        self.model = make_linear_regression()
        self.joint_model = make_linear_regression()

        # Copy over intial weights, so we know they start at the same state.
        self.model.load_state_dict(copy.deepcopy(trust_model.state_dict()))
        self.joint_model.load_state_dict(copy.deepcopy(trust_model.state_dict()))
        self.loss_fn = torch.MSE()

        self.writer = SummaryWriter()

        # Generate Data
        self.train_data, self.val_data, self.test_data = generate_data()

        self.epochs = 10 
        self.lr = 0.01

    def train(self):
        self.train_trust()
        self.train_normal()
    
    def evaluate(self):
        with torch.no_grad():
            for name, model in [("trust", self.trust_model), ("normal", self.model)]:
                model.eval()
                test_predictions = model(self.test_data[DATA_IDX])
                loss = self.loss_fn(test_predictions, self.test_data[GT_IDX])
                print("evaluation for {}: {}".format(name, loss.data))

    def generate_data(dim=1, degree=1, noise=0, num_samples=500):
        """For now just generate data in a single line with no noise.""" 
        # Line is y = 4x + 6 
        # Sample x's uniformly at random from [-100, 100] 
        train_size = int(num_samples * TRAIN_PCT)
        val_size = int(num_samples * VAL_PCT)
        test_size = int(num_samples * TEST_PCT)
        x_samples = np.random.randint(-100, 100, num_samples)
        y_values =  4 * x_samples + 6
        test_start = train_size + val_size
        return (x_values[:train_size], y_values[:train_size]),
            (x_values[train_size:train_size + val_size], y_values[train_size:train_size + val_size]),
            (x_values[test_start: test_start + test_size], y_values[test_start: test_start + test_size])

    @static_method
    def gd_fn_approximation(grad, gamma):
        return (2 + gamma) * grad.dot(grad)

    def train_trust(self):
        optimizer = torch.nn.SGD(self.trust_model.parameters(), lr=self.lr)
        top_validation = 10**10
        gamma = 1
        last_val_error = self.run_validation(self.trust_model)
        top_model = None
        for epoch in epochs:
            predictions = self.trust_model(train_data[DATA_IDX])
            error = self.loss_fn(predictions, train_data[GT_IDX])

            self.writer.add_scalar("trust_region/train", error.data, epoch)
            error.backward()
            val_error = self.run_validation(self.trust_model)

            self.writer.add_scalar("trust_region/val", val_error, epoch)

            d_diff_err = 1
            c_numerator = (last_val_error - val_error)

            # Validation Error Increased, and will be stuck
            if c_numerator < 0:
                print("Validation Increased!")

            c =  c_numerator / float(gd_fn_approximation(model.weights.grad, gamma))
            if c < 1:  # Step is too large, Abort.
                gamma /= 2.0
                optimizer.zero_grad()
            else:
                optimizer.step()

    def train_normal(self):
        optimizer = torch.nn.SGD(self.model.parameters(), lr=self.lr)
        loss_fn = torch.MSE()

        top_validation = 10**10
        top_model = None
        for epoch in epochs:
            predictions = self.model(self.train_data[DATA_IDX])
            error = self.loss_fn(predictions, self.train_data[GT_IDX])
            self.writer.add_scalar("normal/train", error.data, epoch)
            error.backward()
            optimizer.step()
            val_error = run_validation(model, val_data)
            self.writer.add_scalar("normal/val", val_error, epoch)
            if val_error < top_validation:
                top_validation = val_error
                top_model = self.model
        print("top normal model validation: {}".format(validation))
    
    def train_joint(self):
        pass

    def run_validation(self, model):
        with torch.no_grad():
            predictions = model(val_data[DATA_IDX])
            error = self.loss_fn(predictions, val_data[GT_IDX])
            return error.data

    def make_linear_regression(input_dim=1, output_dim=1):
        return torch.nn.Linear(input_dim, output_dim)

    def make_mlp():
        pass


if __name__ == "__main__":
    trust_tester = TrustRegionTester()
    trust_tester.train()
    trust_tester.evaluate()