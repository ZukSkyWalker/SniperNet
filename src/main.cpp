#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include "cnpy.h"
#include "visualize.h"

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <npz_filename>" << std::endl;
    return 1;
  }

  std::string npz_filename = argv[1];

  try {
    cnpy::npz_t npz = cnpy::npz_load(npz_filename);
    cnpy::NpyArray arr;

    // Choose the appropriate variable name from the NPZ file
    std::string variable_name = "pos";
    if (npz.count(variable_name) > 0) {
      arr = npz[variable_name];
    } else {
      std::cerr << "Error: Variable '" << variable_name << "' not found in the NPZ file." << std::endl;
      return 1;
    }

    size_t num_points = arr.shape[0];
    size_t num_dimensions = arr.shape[1];
    
    if (num_dimensions != 3) {
      std::cerr << "Error: The input array must have 3 dimensions per point." << std::endl;
      return 1;
    } 


    std::vector<float> positions(arr.num_vals);

    std::memcpy(positions.data(), arr.data<float>(), arr.num_bytes());

    visualize_points(positions, num_points);
  } catch (const std::runtime_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
