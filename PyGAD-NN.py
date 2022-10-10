#  -*- coding = utf-8 -*- 
#  @time2022/4/1510:40
#  Author:Ruanguojie

import torch
import pygad.torchga
import pygad
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Fitness function
loss_function = torch.nn.MSELoss()
def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function

    predictions = pygad.torchga.predict(model=model,
                                        solution=solution,
                                        data=data_inputs)
    MSE = loss_function(data_outputs,predictions).detach().numpy() + 0.00000001
    # detach().numpy()  change variable to numpy
    solution_fitness = 1.0 / MSE

    return solution_fitness

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

# Create the PyTorch model. Construct neural networks architecture easily.
input_layer = torch.nn.Linear(17, 10)
# change the number of input_features and hidden_unit
activation_layer = torch.nn.Sigmoid()  # activation function
# activation_layer2 = torch.nn.Sigmoid()
hidden_layer = torch.nn.Linear(10, 10)
# dropout_layer = torch.nn.Dropout(p=0.5)   # add dropout layer will reduce model performance in our data
output_layer = torch.nn.Linear(10, 1)
torch.manual_seed(42)
model = torch.nn.Sequential(input_layer,
                            hidden_layer,
                            activation_layer,
                            hidden_layer,
                            activation_layer,
                            hidden_layer,
                            activation_layer,
                            # dropout_layer,
                            output_layer)
# print(model)

# Create an instance of the pygad.torchga. TorchGA class to build the initial population.
torch_ga = pygad.torchga.TorchGA(model=model,
                                 num_solutions=20)
# increase num_solutions will improve model performance sometimes, but increase computation time

# Data inputs
# read data
dataset = pd.read_csv('N/NNI.csv')  # 更换数据
# randomly shuffled data
dataset = dataset.sample(frac=1.0, random_state=42)
data = dataset.reset_index(drop=True)
print(dataset.head())
print(dataset.describe())

# target
target = dataset.pop('NNI')
X_train, X_test, y_train, y_test = train_test_split(dataset.values.astype(np.float32),
                                                    target.values.reshape(-1, 1).astype(np.float32),
                                                    test_size=0.2,
                                                    random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

data_inputs = torch.tensor(X_train_scaled)

# Data outputs
data_outputs = torch.tensor(y_train)

# Prepare the PyGAD parameters.
# Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 500  # Number of generations.
num_parents_mating = 5  # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights  # Initial population of network weights

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation,
                       # save_best_solutions=True,
                       mutation_type='adaptive',  # self-adaptive mutation
                       mutation_probability=np.array([0.25, 0.05])  # mutation probability
                       )

ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)
if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))
# save and load best solution
filename = 'NNI'
ga_instance.save(filename=filename)
loaded_ga_instance = pygad.load(filename=filename)
# print(loaded_ga_instance.best_solution())

# Returning the details of the best solution.
solution, solution_fitness, match_idx = loaded_ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {match_idx}".format(match_idx=match_idx))

predictions = pygad.torchga.predict(model=model,
                                    solution=solution,
                                    data=data_inputs,
                                    )
# print("Predictions : \n", predictions.detach().numpy())

R2 = r2_score(data_outputs.detach().numpy(), predictions.detach().numpy())   # r2_score(test,pred)
MSE = loss_function(data_outputs, predictions)
RMSE = MSE ** 0.5
print("RMSE on training data : ", RMSE.detach().numpy())
print("R^2 on training data : ", R2)

# model evaluation
X_test_scaled = torch.tensor(X_test_scaled)
y_test = torch.tensor(y_test)
predictions_test = pygad.torchga.predict(model=model,
                                         solution=solution,
                                         data=X_test_scaled,
                                         )
# print("Predictions : \n", predictions_test.detach().numpy())
MSE_test = loss_function(y_test, predictions_test)
RMSE_test = MSE_test ** 0.5
R2_test = r2_score(y_test.detach().numpy(), predictions_test.detach().numpy())
# Unlike most other scores,  R2 score may be negative (it need not actually be the square of a quantity R).
print("RMSE on test data: ", RMSE_test.detach().numpy())
print("R^2 on test data: ", R2_test)
plt.plot(predictions_test.detach().numpy(), y_test.detach().numpy(), 'g*')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('$R^{2}$ visual')
plt.show()
# Note that just running the code again may give different results.
