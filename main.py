import csv
import matplotlib.pyplot as plt
def main():
    f_training = open('iris training.csv', 'r')
    f_test = open('iris test.csv', 'r')

    training_reader = csv.reader(f_training, delimiter=';')

    attribute_count = len(next(training_reader))-1
    f_training.seek(0)
    print('number of attributes: ',attribute_count)
    test_reader = csv.reader(f_test, delimiter=';')

    training_data = []
    test_data = []
    # read training data
    for row in training_reader:
        attributes = [float(value) for value in row[:-1]]
        obs = observation(attributes, row[len(row) - 1])
        training_data.append(obs)

    # read test data
    for row in test_reader:
        attributes = [float(value) for value in row[:-1]]
        obs = observation(attributes, row[len(row) - 1])
        test_data.append(obs)

    x_values = []
    y_values = []
    for k in range (1,106):
        k_accuracy = perform_knn(training_data,test_data,k)
        x_values.append(k_accuracy[0])
        y_values.append(k_accuracy[1])


    # Create a scatter plot
    plt.scatter(x_values, y_values, marker='.', color='blue')

    # Add labels and title
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title("k of KNN vs Accuracy")

    # Show the plot
    plt.show()


    k = int(input("Podaj k: "))
    perform_knn(training_data,test_data,k)

    while(True):
        input_vec = []
        for i in range(0,attribute_count):
            input_vec.append(float(input('podaj wartosc atrybutu: ' + str(i+1) + ' ')))
        print(get_most_frequent_species(knn(observation(input_vec,"test"),training_data,k)))
def perform_knn(training_data,test_data,k):

    predicted_correct = 0
    predicted_wrong = 0
    all_obs = len(test_data)
    for obs in test_data:
        neighbors = knn(obs,training_data,k)
        classification = get_most_frequent_species(neighbors)[0]
        if(classification == obs.species):
            predicted_correct += 1
        else:
            predicted_wrong += 1
    #     print('klasyfikacja: ' ,classification,' realne: ', obs.species)
    # print('====================')
    accuracy = predicted_correct/all_obs
    print("k: ",k," accuracy: ",accuracy)
    return (k,accuracy)


class observation:
    def __init__(self, attributes, species):
        self.attributes = attributes
        self.species = species

def get_distance(obs1,obs2):
    sum_squared = 0
    for i in range (0,len(obs1.attributes)):
        sum_squared += (obs1.attributes[i] - obs2.attributes[i])**2
    return (sum_squared ** 0.5)
def knn(test_observation, training_data, k):
    neighbors_distances = []
    for obs in training_data:
        distance = get_distance(obs,test_observation)
        neighbors_distances.append((distance,obs))
    return sorted(neighbors_distances,key =lambda x: [x[0],x[1].species])[:k]

def get_most_frequent_species(neighbors):
    distances_frequencies = {}
    #count how many each species is as a neighbor
    for neighbor in neighbors:
        if (neighbor[1].species in distances_frequencies):
            distances_frequencies[neighbor[1].species] += 1
        else:
            distances_frequencies[neighbor[1].species] = 1
    #return the last pair (species, count) in sorted dictionary -> most frequencies
    return(sorted(distances_frequencies.items(),key = lambda x: [x[1],x[0]])[-1])


if __name__ == "__main__":
    main()