import torch
import numpy as np
import copy
import json
import matplotlib
import matplotlib.pyplot as plt
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
    MAX_GAMMA = 10**8
    MIN_GAMMA = 1
    
    DIM = 10

    def __init__(self):
        # Generate Data
        self.train_data, self.val_data, self.test_data = self.generate_data()
        # self.train_data, self.val_data, self.test_data = self.load_data('data/housing_data.txt')

        # self.find_dataset_distance()

        # Setup models
        self.trust_model = self.make_linear_regression() 
        self.model = self.make_linear_regression()
        self.joint_model = self.make_linear_regression()

        # Copy over intial weights, so we know they start at the same state.
        self.model.load_state_dict(copy.deepcopy(self.trust_model.state_dict()))
        self.joint_model.load_state_dict(copy.deepcopy(self.trust_model.state_dict()))
        self.loss_fn = torch.nn.MSELoss()

        self.writer = SummaryWriter()


        self.epochs = 500
        self.lr = 0.00000001


    def train(self):
        self.train_trust()
        # self.train_normal()

    def graph_num_failures(self):
        sample_ranges = [i+1 for i in range(100)]
        num_trials = 20
        epochs_to_failure = []
        validation_err = []
        test_err = []
        train_err = []
        data_distance = []
        data_distance_2 = []
        for sample_range in sample_ranges:
            for _ in range(num_trials):
                self.train_data, self.val_data, self.test_data = self.generate_data(range=sample_range)
                epochs_to_failure.append(self.train_trust())
                validation_err.append(self.test(self.trust_model, self.val_data))
                test_err.append(self.test(self.trust_model, self.test_data))
                train_err.append(self.test(self.trust_model, self.train_data))
                data_distance.append(float(self.find_dataset_distance()[0]))
                data_distance_2.append(float(self.find_dataset_distance_2()[0]))

        # print(data_distance)
        # print(epochs_to_failure)
        plt.scatter(data_distance_2, epochs_to_failure)
        plt.show()
        """
        plt.scatter(sample_ranges, epochs_to_failure)
        plt.show()
        """
        self.make_bar_graph(50, data_distance_2, epochs_to_failure, self.epochs -1, "Distance 2 Between Validation and Training")
        self.make_bar_graph(50, data_distance, epochs_to_failure, self.epochs - 1, "Distance Between Validation and Training")
        # self.make_bar_graph(50, sample_ranges, epochs_to_failure, self.epochs - 1, "Data Range")
    
    def make_bar_graph(self, num_bins, bin_list, y_list, good_y_thresh, x_label):
        min_val = min(bin_list)
        max_val = max(bin_list)
        bin_size = float(max_val- min_val) / num_bins
        print("bin_size: {}".format(bin_size))
        bin_counts = [0 for i in range(num_bins + 1)]
        bin_finished = [0 for i in range(num_bins + 1)]
        avg_bin = [0 for i in range(num_bins + 1)]

        for idx, value in enumerate(bin_list):
            bin = int((value - min_val)/ bin_size)
            if bin >= len(bin_counts):
                print("overshot bin!")
                continue
            bin_counts[bin]+= 1
            avg_bin[bin] += y_list[idx]
            if y_list[idx] == good_y_thresh:
                bin_finished[bin] += 1

        print("bin_finished: {}".format(bin_finished)) 
        bin_pct_finished = []
        for bin in range(num_bins + 1):
            if bin_counts[bin] > 0:
                bin_finished[bin] /= float(bin_counts[bin])
                avg_bin[bin] /= float(bin_counts[bin])

        # tick_labels = [int(min_val + i * bin_size) for i in range(len(bin_finished))]
        plt.bar([i for i in range(len(bin_finished))], bin_finished) #, tick_label=tick_labels)
        plt.xlabel(x_label)
        plt.ylabel("Pct {} training steps completed".format(self.epochs))
        plt.show()

        tick_labels = [min_val + i * bin_size for i in range(len(avg_bin))]
        plt.bar([i for i in range(len(avg_bin))], avg_bin) # , tick_label=tick_labels)
        plt.xlabel(x_label)
        plt.ylabel("Average number of training steps completed".format(self.epochs))
        plt.show()

    def test(self, model, dataset):
        with torch.no_grad():
            predictions = model(dataset[self.DATA_IDX])
            error = self.loss_fn(predictions, dataset[self.GT_IDX])
            return error.data
        

    def evaluate(self):
        self.find_dataset_distance()
        with torch.no_grad():
            for name, model in [("trust", self.trust_model), ("normal", self.model)]:
                model.eval()
                test_predictions = model(self.test_data[self.DATA_IDX])
                loss = self.loss_fn(test_predictions, self.test_data[self.GT_IDX])
                val_loss = self.test(model, self.val_data)
                print("validation for {}: {}".format(name, val_loss.data))
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
        

    def find_dataset_distance(self):
        total_distance = 0
        print(self.val_data[self.DATA_IDX].shape)
        for val_idx in range(len(self.val_data[self.DATA_IDX])):
            min_distance = 10**10
            for train_idx in range(len(self.train_data[self.DATA_IDX])):
                data_dist = (self.val_data[self.DATA_IDX][val_idx] - self.train_data[self.DATA_IDX][train_idx])
                data_dist = data_dist.dot(data_dist)
                label_dist = 0 #(self.val_data[self.GT_IDX][val_idx] - self.train_data[self.GT_IDX][train_idx])**2
                dist = (data_dist + label_dist)**(1/2.0)
                if dist < min_distance:
                    min_distance = dist
            total_distance += min_distance
        print("dataset distance: {}".format(total_distance))
        return [total_distance]

    def find_dataset_distance_2(self):
        total_distance = 0
        print(self.val_data[self.DATA_IDX].shape)
        for val_idx in range(len(self.val_data[self.DATA_IDX])):
            min_distance = 10**10
            true_distance = 10**10
            for train_idx in range(len(self.train_data[self.DATA_IDX])):
                data_dist = (self.val_data[self.DATA_IDX][val_idx] - self.train_data[self.DATA_IDX][train_idx])
                data_dist = data_dist.dot(data_dist)
                label_dist = (self.val_data[self.GT_IDX][val_idx] - self.train_data[self.GT_IDX][train_idx])**2

                dist = label_dist / float(data_dist + 0.000001)
                if data_dist < min_distance:
                    min_distance = data_dist
                    true_distance = dist
            total_distance += true_distance
        print("dataset distance: {}".format(total_distance))
        return [total_distance]

    def generate_data(self, degree=1, noise=0.0, num_samples=100, range=100):
        """For now just generate data in a single line with no noise.""" 
        # Sample x's uniformly at random from [-100, 100].
        train_size = int(num_samples * self.TRAIN_PCT)
        val_size = int(num_samples * self.VAL_PCT)
        test_size = int(num_samples * self.TEST_PCT)

        # Randomly Generate data.
        x_values = torch.Tensor(np.random.randint(-range, range, (num_samples, self.DIM)))
        weights = torch.Tensor(np.random.randint(-10, 10, (self.DIM, 1)))
        bias = torch.Tensor(np.random.randint(-100, 100, 1))
        print("weights: {}   biases: {}".format(weights, bias))

        # Construct Y values.
        y_values =  torch.mm(x_values, weights) + bias 

        # Add noise.
        y_values = y_values + torch.Tensor(np.random.normal(noise, 0, y_values.shape))
        test_start = train_size + val_size
        # print("x_values: {}".format(x_values))
        # print("y_values: {}".format(y_values))

        return ((x_values[:train_size], y_values[:train_size]),
               (x_values[train_size:train_size + val_size], y_values[train_size:train_size + val_size]),
               (x_values[test_start: test_start + test_size], y_values[test_start: test_start + test_size]))

    def gd_fn_approximation(self, grad, gamma):
        d = self.d(grad, gamma)
        if len(grad.shape) > 1:
            return -(torch.mm(grad, torch.t(d)) + torch.mm(d, torch.t(d)))
        return -(grad * d + d * d)
    
    def d(self, grad, gamma):
        return -float(1 / float(1 + gamma)) *  grad

    def train_trust(self):
        gamma = 1
        last_val_error, grad = self.run_validation(self.trust_model)
        steps_taken = 0
        steps_until_negative_grad_prod = 0
        grad_prod_negative = False
        for epoch in range(self.epochs):
            print("--- Iteration {} ---".format(epoch))
            predictions = self.trust_model(self.train_data[self.DATA_IDX])
            error = self.loss_fn(predictions, self.train_data[self.GT_IDX])
            self.trust_model.zero_grad()

            self.writer.add_scalar("trust_region/train", error.data, epoch)
            error.backward()

            prev_weights = []
            prev_grads = []
            approx_decrease = 0

            for w in self.trust_model.parameters():
                prev_weights.append(w.data.clone())
                prev_grads.append(w.grad.data.clone())
                # w.data = w.data + self.d(w.grad.data, gamma) 
                # approx_decrease += self.gd_fn_approximation(w.grad, gamma)
            
            # To get the gradient of the ORIGINAL model with respect to the validation set
            val_error, val_grads = self.run_validation(self.trust_model)
            
            grad_prod = 0
            for i, prev_grad in enumerate(prev_grads):
                if len(prev_grad.shape) > 1:
                    grad_prod += torch.mm(prev_grad, torch.t(val_grads[i]))
                else: 
                    grad_prod += prev_grad * val_grads[i]

            self.writer.add_scalar("trust_region/grad_prod", grad_prod, epoch)
            print("grad_prod: {}".format(grad_prod))
            if grad_prod < 0:
                print("negative grad_prod!")
                grad_prod_negative = True

            if not grad_prod_negative:
                steps_until_negative_grad_prod += 1

            # Temporarily Take a Step
            for i, w in enumerate(self.trust_model.parameters()):
                prev_weights.append(w.data.clone())
                prev_grads.append(w.grad.data.clone())
                w.data = w.data + self.d(prev_grads[i], gamma) 
                approx_decrease += self.gd_fn_approximation(prev_grads[i], gamma)

            # Check validation on the new step
            val_error, _ = self.run_validation(self.trust_model)

            self.writer.add_scalar("trust_region/val", val_error, epoch)

            c_numerator = (last_val_error - val_error)
            self.writer.add_scalar("trust_region/fn_true", c_numerator, epoch)

            # Validation Error Increased, and will be stuck
            if c_numerator < 0:
                print("Validation Increased!")

            c =  c_numerator / approx_decrease
            self.writer.add_scalar("trust_region/c", c, epoch)
            self.writer.add_scalar("trust_region/fn_approximation", approx_decrease, epoch)
            # print("validation_diff: {}  estimation: {} gamma: {} gdfn_approx: {} prev_grad: {}".format(c_numerator, c, gamma, approx, prev_grad))
            if c < 0:  # Step is too large, Abort.
                # print("c too small, aborting...")
                for idx, w in enumerate(self.trust_model.parameters()):
                    w.data = prev_weights[idx]
                gamma /= self.M_1
                gamma = min(gamma, self.MAX_GAMMA)
                if gamma == self.MAX_GAMMA:
                    print("Epochs to Failure: {} Epochs to negative gradient: {}".format(steps_taken, steps_until_negative_grad_prod))
                    return steps_taken 
            else:
                last_val_error = val_error
                steps_taken += 1
                print("step taken")
                

            if c > self.ETA_3:
                gamma /= self.M_2 
                gamma = max(gamma, self.MIN_GAMMA)
            self.writer.add_scalar("trust_region/gamma", gamma, steps_taken)
        print("Epochs to Failure: {} Epochs to negative gradient: {}".format(steps_taken, steps_until_negative_grad_prod))
        return steps_taken 

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
        top_validation = 10**10
        top_model = None
        data = torch.cat([self.train_data[self.DATA_IDX], self.val_data[self.DATA_IDX]], axis=0)
        gt = torch.cat([self.train_data[self.GT_IDX], self.val_data[self.GT_IDX]], axis=0)
        for epoch in range(self.epochs):
            predictions = self.joint_model(data)
            error = self.loss_fn(predictions, gt)
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

    def run_validation(self, model):
        model.zero_grad()
        predictions = model(self.val_data[self.DATA_IDX])
        error = self.loss_fn(predictions, self.val_data[self.GT_IDX])
        error.backward()
        val_grads = []
        for w in model.parameters():
            val_grad = w.grad.data.clone()
            val_grads.append(val_grad)
        model.zero_grad()
        return error.data, val_grads

    def make_linear_regression(self, output_dim=1):
        layer = torch.nn.Linear(self.DIM, output_dim)
        torch.nn.init.constant_(layer.weight, 0)
        torch.nn.init.constant_(layer.bias , 0)
        return layer

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
    # trust_tester.graph_num_failures()
    trust_tester.train()
    trust_tester.evaluate()