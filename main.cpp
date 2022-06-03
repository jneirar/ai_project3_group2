#include "MLP.cpp"

int main(){
    bnu::matrix<double> x_train(3, 3); //3 datos con 3 características
    bnu::matrix<double> y_train(3, 2); //2 clases
    bnu::matrix<double> x_validation(2, 3); //2 datos con 3 características
    bnu::matrix<double> y_validation(2, 2); //2 clases
    x_train <<= 1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0;
    y_train <<= 1.0, 0.0,
                0.0, 1.0,
                0.0, 1.0;

    bnu::vector<int> neuronas_by_layer(2); //2 capas
    neuronas_by_layer <<= 10, 15;
    std::string activation_function_name = "sigmoid";
    
    MLP mlp(neuronas_by_layer, activation_function_name);
    int epoch = 1000;
    double alpha = 0.01;
    mlp.train(x_train, y_train, x_validation, y_validation, epoch, alpha, false);

    return 0;
}