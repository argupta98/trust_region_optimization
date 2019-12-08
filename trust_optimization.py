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
    
    DIM = 1 

    def __init__(self):
        # Generate Data
        self.train_data, self.val_data, self.test_data = self.generate_data()
        # self.train_data, self.val_data, self.test_data = self.load_data('data/housing_data.txt')

        # Setup models
        self.trust_model = self.make_linear_regression() 
        self.model = self.make_linear_regression()
        self.joint_model = self.make_linear_regression()

        # Copy over intial weights, so we know they start at the same state.
        self.model.load_state_dict(copy.deepcopy(self.trust_model.state_dict()))
        self.joint_model.load_state_dict(copy.deepcopy(self.trust_model.state_dict()))
        self.loss_fn = torch.nn.MSELoss()

        self.writer = SummaryWriter()


        self.epochs = 10000
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

    def load_data(self, data_path):
        with open(data_path, 'r') as fp:
            lines = fp.readlines()
            data_tensor = []
            gt_tensor = []
            for line in lines:
                data_points = line.split(' ')
                cleaned_data = []
                for i in range(len(data_points)):
                    stripped_data = data_points[i].strip()
                    if len(stripped_data) > 0:
                        cleaned_data.append(float(stripped_data))
                gt_tensor.append([cleaned_data[-1]])
                data_tensor.append(cleaned_data[:-1])
                self.DIM = len(data_tensor[0])

        num_samples = len(data_tensor)
        train_size = int(num_samples * self.TRAIN_PCT)
        val_size = int(num_samples * self.VAL_PCT)
        test_size = int(num_samples * self.TEST_PCT)
        data_tensor = torch.Tensor(data_tensor)
        gt_tensor = torch.Tensor(gt_tensor)

        test_start = train_size + val_size
        return ((data_tensor[:train_size], gt_tensor[:train_size]),
               (data_tensor[train_size:train_size + val_size], gt_tensor[train_size:train_size + val_size]),
               (data_tensor[test_start: test_start + test_size], gt_tensor[test_start: test_start + test_size]))
        


    def generate_data(self, degree=1, noise=0.0, num_samples=500):
        """For now just generate data in a single line with no noise.""" 
        # Sample x's uniformly at random from [-100, 100].
        train_size = int(num_samples * self.TRAIN_PCT)
        val_size = int(num_samples * self.VAL_PCT)
        test_size = int(num_samples * self.TEST_PCT)

        # Randomly Generate data.
        x_values = torch.Tensor(np.random.randint(-100, 100, (num_samples, self.DIM)))
        weights = torch.Tensor(np.random.randint(-10, 10, (self.DIM, 1)))
        bias = torch.Tensor(np.random.randint(-100, 100, 1))
        print("weights: {}   biases: {}".format(weights, bias))

        # Construct Y values.
        y_values =  torch.mm(x_values, weights) + bias 

        # Add noise.
        y_values = y_values + torch.Tensor(np.random.normal(noise, 0, y_values.shape))
        x_values = x_values
        test_start = train_size + val_size

        return ((x_values[:train_size], y_values[:train_size]),
               (x_values[train_size:train_size + val_size], y_values[train_size:train_size + val_size]),
               (x_values[test_start: test_start + test_size], y_values[test_start: test_start + test_size]))

    def gd_fn_approximation(self, grad, gamma):
        d = self.d(grad, gamma)
        return -(torch.mm(grad, torch.t(d)) + torch.mm(d, torch.t(d)))
    
    def d(self, grad, gamma):
        return -float(1 / float(1 + gamma)) *  grad

    def train_trust(self):
        gamma = 1
        last_val_error = self.run_validation(self.trust_model)
        for epoch in range(self.epochs):
            # print("--- Iteration {} ---".format(epoch))
            predictions = self.trust_model(self.train_data[self.DATA_IDX])
            error = self.loss_fn(predictions, self.train_data[self.GT_IDX])
            self.trust_model.zero_grad()
            self.writer.add_scalar("trust_region/train", error.data, epoch)
            error.backward()

            # Temporarily Take a Step
            prev_grad = self.trust_model.weight.grad.clone()
            prev_bias_grad = self.trust_model.bias.grad.clone()
            prev_weights = self.trust_model.weight.clone()
            prev_bias = self.trust_model.bias.clone()
            self.trust_model.weight = torch.nn.Parameter(self.trust_model.weight + self.d(prev_grad, gamma))
            self.trust_model.bias = torch.nn.Parameter(self.trust_model.bias + self.d(prev_bias_grad, gamma))

            val_error = self.run_validation(self.trust_model)

            self.writer.add_scalar("trust_region/val", val_error, epoch)

            c_numerator = (last_val_error - val_error)
            # self.writer.add_scalar("trust_region/val_diff", c_numerator, epoch)

            # Validation Error Increased, and will be stuck
            if c_numerator < 0:
                print("Validation Increased!")


            approx = float(self.gd_fn_approximation(prev_grad, gamma))
            c =  c_numerator / approx
            # print("validation_diff: {}  estimation: {} gamma: {} gdfn_approx: {} prev_grad: {}".format(c_numerator, c, gamma, approx, prev_grad))
            if c < self.ETA_1:  # Step is too large, Abort.
                #print("c too small, aborting...")
                self.trust_model.weight = torch.nn.Parameter(prev_weights)
                self.trust_model.bias = torch.nn.Parameter(prev_bias)
            else:
                last_val_error = val_error
                

            if c < self.ETA_2:
                gamma /= self.M_1
            elif c > self.ETA_3:
                gamma /= self.M_2 
            self.writer.add_scalar("trust_region/gamma", gamma, epoch)

    def train_normal(self):
        top_validation = 10**10
        top_model = None
        for epoch in range(self.epochs):
            predictions = self.model(self.train_data[self.DATA_IDX])
            error = self.loss_fn(predictions, self.train_data[self.GT_IDX])
            self.writer.add_scalar("normal/train", error.data, epoch)
            self.model.zero_grad()
            error.backward()
            for w in self.model.parameters():
                w.data = w.data - self.lr * w.grad.data

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

    def make_linear_regression(self, output_dim=1):
        return torch.nn.Linear(self.DIM, output_dim)

    def make_mlp(self, hidden_dims=[300, 200, 100]):
        return MLP(self.DIM, hidden_dims)

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super().__init__()
        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_sizes[0]))
        layers.append(torch.nn.ReLU())
        for idx in range(len(hidden_sizes) - 1):
            curr_size = hidden_sizes[idx]
            next_size = hidden_sizes[idx + 1]
            layers.append(torch.nn.Linear(curr_size, next_size, bias=True))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_sizes[-1], 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, xb):
        return self.network(xb)

if __name__ == "__main__":
    trust_tester = TrustRegionTester()
    trust_tester.train()
    trust_tester.evaluate()