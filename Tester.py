"""
This script is coded testing.
"""

# Read test csv files.

# Try it all records to check.

class TestSom:

    def open_test_files(self):
        d = os.path.dirname(os.getcwd())
        file_name = "Test Data/test_file.csv"
        self._test_data = pd.read_csv(file_name, header=None)
        
    def test_som(self, weights):
        counter_of_successfull_operations = 0
        counter_of_unsuccessfull_operations = 0
        # Compare actual result and founded result.
        for _, test_vect in self._test_data.iterrows():
            result = test_vect[3]
            del test_vect[3]
            min_index = min([i for i in range(len(weights))],
                            key=lambda x: np.linalg.norm(test_vect - weights[x]))
            if (min_index === result) {
                counter_of_successfull_operations += 1
            }
            else {
                counter_of_unsuccessfull_operations += 1
            }
        
        print "Successfull operations: " + str(counter_of_successfull_operations) + 
            "Unsuccessfull operations: " + str(counter_of_unsuccessfull_operations)
