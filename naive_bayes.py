import sys

class NaiveBayes:
    trained = False
    attr_names = None
    common_distribution = None
    distribution = None
    instances = None
    outcome_name = None
    
    def _assert_trained(self):
        assert self.trained, "Model has not been trained yet"
    
    def __init__(self, attr_names = None, instances = None) -> None:
        if attr_names and instances:
            self.train(attr_names, instances)

    def train(self, attr_names, labeled_instances) -> None:
        def add_to_dict(d: dict, key: any, val: int) -> None:
            if key not in d: d[key] = val
            else: d[key] += val
        self.attr_names = attr_names[:]
        self.instances = labeled_instances[:]
        self.outcome_name = self.attr_names.pop()
        attr_count = len(self.attr_names)
        self.distribution = [dict() for i in range(attr_count+1)]
        self.common_distribution = [dict() for i in range(attr_count)]
        for instance in self.instances:            
            for i in range(attr_count):
                add_to_dict(self.distribution[i], instance[i], 1)
                add_to_dict(self.common_distribution[i], (instance[i], instance[attr_count]), 1)
            add_to_dict(self.distribution[attr_count], instance[attr_count], 1)
        self.trained=True
        # distribution and common_distribution are instance counts and not probabilities
    
    def get_probabilities(self, attr_names, instances, laplace_smoothing = False):
        self._assert_trained()
        if attr_names != self.attr_names:
            print("Attribute names in the test and train files are different")
            print(f"Expected {self.attr_names}, found {attr_names}")
            exit(0)
        
        attr_count = len(attr_names)
        predictions = []
        for instance in instances:
            sum_outcomes = 0
            est_outcomes = dict()
            for (outcome_name, outcome_count) in self.distribution[attr_count].items():
                total_cond_prob = outcome_count
                for i in range(attr_count):
                    if laplace_smoothing:
                        total_cond_prob *= (self.common_distribution[i].get((instance[i], outcome_name),0)+1) / (outcome_count + attr_count)
                    else:
                        total_cond_prob *= self.common_distribution[i].get((instance[i], outcome_name),0) / outcome_count
                est_outcomes[outcome_name] = total_cond_prob
                sum_outcomes += total_cond_prob

            assert sum_outcomes, f"All outcomes for instance {instance} have a probability of 0"
            for outcome_name in est_outcomes.keys():
                est_outcomes[outcome_name] /= sum_outcomes
            
            predictions.append(est_outcomes)
        return predictions

    def get_predictions(self, attr_names, instances, laplace_smoothing = False):
        predictions = self.get_probabilities(attr_names, instances, laplace_smoothing)
        outcomes = []
        for prediction in predictions:
            max_prob = max(prediction.values())
            outcomes.append(next(outcome for (outcome, prob) in prediction.items() if prob == max_prob))
        return outcomes
    
    @staticmethod
    def training_error(model: NaiveBayes, laplace_smoothing = False):
        model._assert_trained()
        predictions = model.get_predictions(model.attr_names, model.instances, laplace_smoothing=False)
        n = len(model.instances)
        attr_count = len(model.attr_names)
        return sum(model.instances[i][attr_count] != predictions[i] for i in range(n))
    
    @staticmethod
    def cvloo_error(model: NaiveBayes, laplace_smoothing = False):
        model._assert_trained()
        n = len(model.instances)
        error_count = 0
        for exclude in range(n):
            labeled_instances_exclude_one = model.instances[:]
            deleted_instance = model.instances[exclude][:]
            expected_label = deleted_instance.pop()
            del labeled_instances_exclude_one[exclude]
            new_model = NaiveBayes(model.attr_names + [model.outcome_name], labeled_instances_exclude_one)
            error_count += new_model.get_predictions(model.attr_names, [deleted_instance])[0] != expected_label
        return error_count

if len(sys.argv) != 3:
    print("Usage: python3 naive_bayes.py [train csv path] [test csv path]")
    exit(0)

def parse_csv(path: str):
    with open(path,"r") as fin:
        first = True
        body = []
        for row in fin:
            trimmed_row = [token.strip() for token in row.split(',')]
            if first: 
                header = trimmed_row
                first = False
            else: body.append(trimmed_row)
    return header, body


train_data = parse_csv(sys.argv[1])
test_data = parse_csv(sys.argv[2])
model = NaiveBayes(*train_data)
print(model.get_probabilities(*test_data, laplace_smoothing=True))
print(model.get_predictions(*test_data, laplace_smoothing=True))
print(f"Training error: {NaiveBayes.training_error(model, laplace_smoothing=False)}/{len(train_data[1])}")
print(f"CVLOO error: {NaiveBayes.cvloo_error(model, laplace_smoothing=False)}/{len(train_data[1])}")