#include "MLP.cpp"
bnu::matrix<double> x_train, y_train, x_validation, y_validation, x_test, y_test;

std::string file_input, file_output;

void read(){
    //read matrixes
}

int main(int argc, char *argv[]){
    if(argc < 4) {
        std::cout << "Too low arguments. Usage: ./main <input_file> <output_file> <layers> ";
        std::cout << "<neurons_in_layer(1)1> <neurons_in_layer(2)> ... <neurons_in_layer(layers)> ";
        std::cout << "<activation_function> <epoch> <alpha> <debug>\n";
        return 0;
    }
    file_input = argv[1];
    file_output = argv[2];
    int layers = atoi(argv[3]);
    int arguments = 4 + layers + 4;
    if(argc < arguments){
        std::cout << "Few arguments. Usage: ./main <input_file> <output_file> <layers> ";
        std::cout << "<neurons_in_layer(1)1> <neurons_in_layer(2)> ... <neurons_in_layer(layers)> ";
        std::cout << "<activation_function> <epoch> <alpha> <debug>\n";
        return 0;
    }
    if(argc > arguments){
        std::cout << "Many arguments. Usage: ./main <input_file> <output_file> <layers> ";
        std::cout << "<neurons_in_layer(1)1> <neurons_in_layer(2)> ... <neurons_in_layer(layers)> ";
        std::cout << "<activation_function> <epoch> <alpha> <debug>\n";
        return 0;
    }

    bnu::vector<int> neuronas_by_layer(layers);
    for(int i = 0; i < layers; i++)
        neuronas_by_layer(i) = atoi(argv[4+i]);

    MLP mlp(neuronas_by_layer);

    std::string activation_function_name = argv[4+layers];
    if(activation_function_name != "sigmoid" && activation_function_name != "relu" && activation_function_name != "tanh"){
        std::cout << "Activation function not supported\n";
        std::cout << "Supported functions: sigmoid, relu, tanh\n";
        return 0;
    }
    int epoch = atoi(argv[5+layers]);
    if(epoch < 1)
        epoch = 1;
    double alpha = atof(argv[6+layers]);
    if(alpha < 1e-20)
        alpha = 1e-20;
    std::string debug_str = argv[7+layers];
    if(debug_str != "1" && debug_str != "0"){
        std::cout << "Debug value not supported\n";
        std::cout << "Supported values: 0, 1\n";
        return 0;
    }
    bool debug = debug_str == "1";
    read();
    mlp.train(x_train, y_train, x_validation, y_validation, epoch, alpha, activation_function_name, debug);
    mlp.predict(x_test, y_test);
    mlp.write_errors(file_output);
    return 0;
}