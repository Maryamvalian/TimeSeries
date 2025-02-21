import numpy as np

# Function to calculate coefficients
def find_coefficients(v0, v1, v2, v3, y):
    # Create a matrix of the vectors
    matrix = np.array([v0, v1, v2, v3]).T  # Transpose to align vectors as columns
    # Solve the system of equations to find the coefficients
    coefficients = np.linalg.solve(matrix, y)
    return coefficients

# Input: receiving vectors as input
def get_vector_input(name):
    return list(map(float, input(f"Enter {name} vector (4 float numbers separated by spaces): ").split()))

# Main program to execute the logic
def main():
    # Receive input vectors
    v0 = get_vector_input("v0")
    v1 = get_vector_input("v1")
    v2 = get_vector_input("v2")
    v3 = get_vector_input("v3")
    y = get_vector_input("y")

    # Calculate the coefficients
    coefficients = find_coefficients(v0, v1, v2, v3, y)

    # Output the coefficients
    print("\nThe coefficients are:")
    print(f"c0 = {coefficients[0]}")
    print(f"c1 = {coefficients[1]}")
    print(f"c2 = {coefficients[2]}")
    print(f"c3 = {coefficients[3]}")

if __name__ == "__main__":
    main()
