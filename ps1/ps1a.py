"""
Created on June 9, 2020

@author: Hayk Stepanyan
"""

from ps1_partition import get_partitions
import time


# ================================
# Part A: Transporting Space Cows
# ================================

# Problem 1
def load_cows(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and return a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """
    cow_data = open(filename, "r")
    cow_dict = {}
    for line in cow_data:
        line = line.strip()
        cow_dict.update({line.split(",")[0]: line.split(",")[1]})
    return cow_dict


# Problem 2
def greedy_cow_transport(cows, limit=10):
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    cow_dict_copy = {}
    cow_dict_copy.update(sorted(cows.items(), key=lambda x: x[1], reverse=True))

    trips = []
    while len(cow_dict_copy) != 0:
        trip, added_cow = [], []
        total_cost = 0
        for cow in cow_dict_copy.keys():
            if int(cow_dict_copy[cow]) + total_cost <= limit:
                trip.append(cow)
                total_cost += int(cow_dict_copy[cow])
                added_cow.append(cow)
        for cow in added_cow:
            del cow_dict_copy[cow]
        trips.append(trip)
    return trips


# Problem 3
def brute_force_cow_transport(cows, limit=10):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips 
        Use the given get_partitions function in ps1_partition.py to help you!
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation
            
    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)
    
    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    best_count = 10
    best_trip = []
    cow_list = sorted(cows.items(), key=lambda x: x[1], reverse=True)
    for partition in get_partitions(cow_list):
        allowed_trip_count = 0
        for trip in range(len(partition)):
            total_cost = 0
            for i in range(len(partition[trip])):
                total_cost += int(partition[trip][i][1])
            if total_cost <= limit:
                allowed_trip_count += 1
        if allowed_trip_count == len(partition):
            if len(partition) <= best_count:
                best_count = len(partition)
                best_trip = partition
    return best_trip


# Problem 4
def compare_cow_transport_algorithms():
    """
    Using the data from ps1_cow_data.txt and the specified weight limit, run your
    greedy_cow_transport and brute_force_cow_transport functions here. Use the
    default weight limits of 10 for both greedy_cow_transport and
    brute_force_cow_transport.
    
    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    cows = load_cows("ps1_cow_data.txt")
    start_greedy = time.time()
    print(greedy_cow_transport(cows, limit=10))
    end_greedy = time.time()
    print("Time spent on Greedy Algorithm:", end_greedy - start_greedy)

    start_brute_force = time.time()
    print(brute_force_cow_transport(cows, limit=10))
    end_brute_force = time.time()
    print("Time spent on Brute Force Algorithm:", end_brute_force - start_brute_force)


if __name__ == '__main__':
    compare_cow_transport_algorithms()
