#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <iomanip>
#include <math.h>
#include <fstream>

#include <boost/random/random_device.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/assignment.hpp>

namespace bnu = boost::numeric::ublas;

double sigmoid(double x){
    return 1.0/(1.0+exp(-x));
}

double relu(double x){
    return x > 0 ? x : 0.0;
}

double tanhiperbolica(double x){
    return tanh(x);
}

double square(double x){
    return x*x;
}

double exponencial(double x){
    return exp(x);
}

class MLP{
private:
    int n, m, layers, classes, Ws_size;
    bnu::vector<int> neuronas_by_layer;
    std::vector<bnu::matrix<double>> Ws;
    std::vector<int> Ws_rows, Ws_columns;
    bnu::matrix<double> x_train, y_train, x_validation, y_validation;
    bnu::matrix<double> out_o, out_o_softmax;
    std::vector<bnu::matrix<double>> out_neurons;
    double alpha;
    std::string activation_function_name;
    std::vector<double> errors_training, errors_validation;

    bnu::vector<int> input_model, output_model, output_model_softmax;
    bnu::matrix<double> output_model_softmax_matrix;

    bnu::matrix<double> simple_product(bnu::matrix<double> &m1, bnu::matrix<double> &m2);
    bnu::matrix<double> simple_product(bnu::matrix<double> &m, bnu::vector<double> &v);
    bnu::matrix<double> simple_product(bnu::matrix<double> &m, bnu::matrix_row<bnu::matrix<double>> &v);
    bnu::matrix<double> simple_product(bnu::matrix_row<bnu::matrix<double>> &v1, bnu::matrix_row<bnu::matrix<double>> &v2);
    bnu::matrix<double> simple_product(bnu::vector<double> &v, bnu::matrix_row<bnu::matrix<double>> &mr);
    
    void fill_w(bnu::matrix<double> &m);
    void forward_propagation(bool for_training ,bool debug);
    void backward_propagation(bool debug);
    void error(bool debug);
    
public:
    MLP(bnu::vector<int> &neuronas_by_layer);
    ~MLP();
    void train(bnu::matrix<double> &x_train, bnu::matrix<double> &y_train, bnu::matrix<double> &x_validation, bnu::matrix<double> &y_validation, int epochs, double alpha, std::string activation_function_name, bool debug);
    void predict(bnu::matrix<double> &x_test, bnu::matrix<double> &y_test);
    void predict_one(bnu::vector<double> &input_to_predict);
    void write_errors(std::string filename);
};

void MLP::fill_w(bnu::matrix<double> &m){
    std::random_device rd;
    std::default_random_engine generator(rd()); // rd() provides a random seed
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < m.size1(); i++)
        for (int j = 0; j < m.size2(); j++)
            m(i, j) = distribution(generator);
}

bnu::matrix<double> MLP::simple_product(bnu::matrix<double> &m1, bnu::matrix<double> &m2){
    bnu::matrix<double> m3(m1.size1(), m1.size2());
    for(int i = 0; i < m1.size1(); i++)
        for(int j = 0; j < m1.size2(); j++)
            m3(i, j) = m1(i, j) * m2(i, j);
    return m3;
}

bnu::matrix<double> MLP::simple_product(bnu::matrix<double> &m, bnu::vector<double> &v){
    //m debe ser de dimensiones (1, v.size())
    bnu::matrix<double> m_r(1, v.size());
    for(int i = 0; i < v.size(); i++)
        m_r(0, i) = v(i) * m(0, i);
    return m_r;
}

bnu::matrix<double> MLP::simple_product(bnu::matrix<double> &m, bnu::matrix_row<bnu::matrix<double>> &v){
    //m debe ser de dimensiones (1, v.size())
    bnu::matrix<double> m_r(1, v.size());
    for(int i = 0; i < v.size(); i++)
        m_r(0, i) = v(i) * m(0, i);
    return m_r;
}

bnu::matrix<double> MLP::simple_product(bnu::matrix_row<bnu::matrix<double>> &v1, bnu::matrix_row<bnu::matrix<double>> &v2){
    //v1 y v2 deben ser del mismo tamaño
    bnu::matrix<double> m_r(1, v1.size());
    for(int i = 0; i < v1.size(); i++)
        m_r(0, i) = v1(i) *v2(i);
    return m_r;
}

bnu::matrix<double> MLP::simple_product(bnu::vector<double> &v, bnu::matrix_row<bnu::matrix<double>> &mr){
    //v y mr deben ser del mismo tamaño
    bnu::matrix<double> m_r(1, v.size());
    for(int i = 0; i < v.size(); i++)
        m_r(0, i) = v(i) * mr(i);
    return m_r;
}

void MLP::forward_propagation(bool for_training, bool debug = false){
    if(for_training){
        if(debug) std::cout << "\n-------------Forward propagation------------\n";
        for(int i = 0; i < this->n; i++){
            bnu::matrix_row<bnu::matrix<double>> data_i(this->x_train, i);
            bnu::vector<double> out_layer(data_i);
            if(debug) std::cout << "input " << i << ": " << data_i << "\n";
            //Para cada matriz, iniciamos desde el dato (data_i) y terminamos con un out_o (output obtenido)
            //Se guarda la salida de cada neura de cada capa
            for(int i_matrix = 0; i_matrix < this->Ws_size; i_matrix++){
                //Define el input de la matriz (1 al inicio por el bias)
                bnu::vector<double> input_layer(out_layer.size() + 1);
                input_layer <<= 1, out_layer;
                if(debug) std::cout << "For matrix " << i_matrix << ":\n";
                if(debug) std::cout << "\tinput: " << input_layer << "\n";
                
                //Producto input x matriz
                out_layer = bnu::prod(input_layer, Ws[i_matrix]);
                if(debug) std::cout << "\toutput: " << out_layer << "\n";
                
                //Aplicamos la función de activación
                if(this->activation_function_name == "sigmoid")
                    boost::range::transform(out_layer, out_layer.begin(), sigmoid);
                else if(this->activation_function_name == "relu")
                    boost::range::transform(out_layer, out_layer.begin(), relu);
                else if(this->activation_function_name == "tanh")
                    boost::range::transform(out_layer, out_layer.begin(), tanhiperbolica);
                else //Sigmoid por defecto
                    boost::range::transform(out_layer, out_layer.begin(), sigmoid);
                if(debug) std::cout << "\tact_funct(out): " << out_layer << "\n";

                //Guardamos la salida de cada capa (la última salida no corresponde a alguna capa de la red)
                if(i_matrix < this->Ws_size - 1)
                    for(int j = 0; j < out_layer.size(); j++)
                        out_neurons[i_matrix](i, j) = out_layer(j);
            }
            //Guardamos la salida del dato en la matriz de salida
            for(int j = 0; j < out_layer.size(); j++) this->out_o(i, j) = out_layer(j);
            //Aplicamos la función softmax en out_o
            bnu::vector<double> out_layer_softmax(out_layer);
            boost::range::transform(out_layer_softmax, out_layer_softmax.begin(), exponencial);
            double denominator = boost::accumulate(out_layer_softmax, 0.0);
            out_layer_softmax /= denominator;
            //Guardamos la salida softmax del dato en la matriz de salida softmax
            for(int j = 0; j < out_layer_softmax.size(); j++) this->out_o_softmax(i, j) = out_layer_softmax(j);
            if(debug) std::cout << "output " << i << ": " << out_layer << "\n";
            if(debug) std::cout << "output softmax " << i << ": " << out_layer_softmax << "\n";
        }
        if(debug) std::cout << "\n-----------Forward propagation End----------\n";
    }else{
        //Forward para input_model
        bnu::vector<double> out_layer(this->input_model);
        if(debug) std::cout << "input model = " << this->input_model << "\n";
        
        for(int i_matrix = 0; i_matrix < this->Ws_size; i_matrix++){
            //Define el input de la matriz (1 al inicio por el bias)
            bnu::vector<double> input_layer(out_layer.size() + 1);
            input_layer <<= 1, out_layer;
            if(debug) std::cout << "\tinput: " << input_layer << "\n";
            
            //Producto input x matriz
            out_layer = bnu::prod(input_layer, Ws[i_matrix]);
            if(debug) std::cout << "\toutput: " << out_layer << "\n";
            
            //Aplicamos la función de activación
            if(this->activation_function_name == "sigmoid")
                boost::range::transform(out_layer, out_layer.begin(), sigmoid);
            else if(this->activation_function_name == "relu")
                boost::range::transform(out_layer, out_layer.begin(), relu);
            else if(this->activation_function_name == "tanh")
                boost::range::transform(out_layer, out_layer.begin(), tanhiperbolica);
            else //Sigmoid por defecto
                boost::range::transform(out_layer, out_layer.begin(), sigmoid);
            if(debug) std::cout << "\tact_funct(out): " << out_layer << "\n";
        }

        this->output_model = out_layer;
        //Aplicamos la función softmax en output_model
        this->output_model_softmax = this->output_model;
        boost::range::transform(this->output_model_softmax, this->output_model_softmax.begin(), exponencial);
        double denominator = boost::accumulate(this->output_model_softmax, 0.0);
        this->output_model_softmax /= denominator;
        
        if(debug) std::cout << "output = " << this->output_model << "\n";
        if(debug) std::cout << "output softmax = " << this->output_model_softmax << "\n";
    }
}

void MLP::backward_propagation(bool debug = false){
    if(debug) std::cout << "\n------------Backward propagation------------\n";
    //Cálculo de las derivadas respecto a W, se suma la derivada de cada dato //TODO: No es por lote
    std::vector<bnu::matrix<double>> Ws_derivades(Ws.size());
    for(int i = 0; i < Ws.size(); i++)
        Ws_derivades[i] = bnu::matrix<double>(this->Ws[i].size1(), this->Ws[i].size2(), 0.0);
    
    std::vector<bnu::matrix<double>> deltas(n);
    bnu::matrix<double> delta;
    bnu::vector<double> delta_vector; //Temporal para el cálculo de delta
    
    for(int i_matrix = this->Ws_size - 1; i_matrix >= 0; i_matrix--){
        if(debug) std::cout << "In matrix " << i_matrix << ":\n";
        for(int i_data = 0; i_data < this->n; i_data++){
            if(debug) std::cout << "- data " << i_data << ":\n";
            
            //Si es la última capa
            if(i_matrix == this->Ws_size - 1){
                bnu::vector<double> ones_softmax(this->classes, 1.0);
                bnu::matrix_row<bnu::matrix<double>> out_d_i_vector(this->y_train, i_data), 
                                                    out_o_i_vector(this->out_o, i_data), 
                                                    out_o_softmax_i_vector(this->out_o_softmax, i_data);
                //(S'o - Sd)
                delta_vector = out_o_softmax_i_vector - out_d_i_vector;
                //(S'o - Sd) * (S'o)
                delta = simple_product(delta_vector, out_o_softmax_i_vector);
                //(1 - S'o)
                ones_softmax -= out_o_softmax_i_vector;
                //(S'o - Sd) * (S'o) * (1 - S'o)
                delta = simple_product(delta, ones_softmax);
                if(this->activation_function_name == "sigmoid"){
                    //Derivada de la función de activación: (1 - So) * (So)
                    //(S'o - Sd) * (S'o) * (1 - S'o) * [ (So) ]
                    bnu::vector<double> ones(this->classes, 1.0);
                    ones -= out_o_i_vector;
                    delta = simple_product(delta, out_o_i_vector);
                    //(S'o - Sd) * (S'o) * (1 - S'o) * [ (So) * (1 - So) ]        [] -> derivada de la función de activación
                    delta = simple_product(delta, ones);
                }else if(this->activation_function_name == "tanh"){
                    //Derivada de la función de activación: (1 - So) * (1 + So)
                    //(S'o - Sd) * (S'o) * (1 - S'o) * [ (1 - So) ]
                    bnu::vector<double> ones_minus(this->classes, 1.0);
                    ones_minus -= out_o_i_vector;
                    delta = simple_product(delta, ones_minus);
                    //(S'o - Sd) * (S'o) * (1 - S'o) * [ (1 - So) * (1 + So) ]
                    bnu::vector<double> ones_plus(this->classes, 1.0);
                    ones_plus -= out_o_i_vector;
                    delta = simple_product(delta, ones_plus);
                }else if(this->activation_function_name == "relu"){
                    //Derivada de la función de activación: 0 si So < 0, 1 si So > 0
                    bnu::vector<double> mask(this->classes, 0.0);
                    for(int i = 0; i < this->classes; i++)
                        if(out_o_i_vector(i) > 0) mask(i) = 1.0;
                    delta = simple_product(delta, mask);
                }else{//Sigmoid por defecto
                    //(S'o - Sd) * (S'o) * (1 - S'o) * [ (So) ]
                    bnu::vector<double> ones(this->classes, 1.0);
                    ones -= out_o_i_vector;
                    delta = simple_product(delta, out_o_i_vector);
                    //(S'o - Sd) * (S'o) * (1 - S'o) * [ (So) * (1 - So) ]        [] -> derivada de la función de activación
                    delta = simple_product(delta, ones);
                }
                
                if(debug) std::cout << "\tdelta = " << delta << "\n";
                
                //Guardo el delta para la siguiente matriz
                deltas[i_data] = delta;
                
                //(Si) -> Salida de la última capa correspondiente a i_data (1 al inicio por el bias)
                bnu::matrix_row<bnu::matrix<double>> out_neurons_i_vector(this->out_neurons[i_matrix - 1], i_data);
                bnu::matrix<double> out_neurons_i(out_neurons_i_vector.size() + 1, 1);
                out_neurons_i <<= 1, out_neurons_i_vector;
                if(debug) std::cout << "\tout_neurons_i = " << out_neurons_i << "\n";
                
                //Derivada = Si * delta, da una matriz del mismo tamaño que Wi
                bnu::matrix<double> derivadas_0 = bnu::prod(out_neurons_i, delta);
                if(debug) std::cout << "\tderivadas_ " << i_matrix << " = " << derivadas_0 << "\n";
                //Acumulo la derivada
                Ws_derivades[i_matrix] += derivadas_0;
            }else{
                //Extrae Ws de la sgte capa menos la primera fila
                bnu::matrix<double> Ws_next(this->Ws[i_matrix + 1].size1() - 1, this->Ws[i_matrix + 1].size2());
                for(int i = 0; i < Ws_next.size1(); i++)
                    for(int j = 0; j < Ws_next.size2(); j++)
                        Ws_next(i, j) = this->Ws[i_matrix + 1](i + 1, j);
                if(debug) std::cout << "\tWs_next = " << Ws_next << "\n";
                
                //Extrae el prev delta correspondiente al dato i_data
                delta = deltas[i_data];
                if(debug) std::cout << "\tdelta = " << delta << "\n";
                //Sum de delta * w de la capa previa:
                delta = bnu::prod(Ws_next, bnu::trans(delta));
                delta = bnu::trans(delta);
                if(debug) std::cout << "\tdelta x w = " << delta << "\n";
                
                //Extrae Sj, salida de la capa de salida de la matriz
                bnu::matrix_row<bnu::matrix<double>> out_neurons_j_vector(this->out_neurons[i_matrix], i_data);
                if(debug) std::cout << "\tout_neurons_j = " << out_neurons_j_vector << "\n";

                if(this->activation_function_name == "sigmoid"){
                    //New delta: delta * [ (Sj) * (1 - Sj) ]  [] -> derivada de la función de activación
                    delta = simple_product(delta, out_neurons_j_vector);
                    bnu::vector<double> ones(out_neurons_j_vector.size(), 1.0);
                    ones -= out_neurons_j_vector;
                    delta = simple_product(delta, ones);
                }else if(this->activation_function_name == "tanh"){
                    //New delta: delta * [ (1 + Sj) * (1 - Sj) ]  [] -> derivada de la función de activación
                    bnu::vector<double> ones_minus(out_neurons_j_vector.size(), 1.0);
                    ones_minus -= out_neurons_j_vector;
                    delta = simple_product(delta, ones_minus);
                    bnu::vector<double> ones_plus(out_neurons_j_vector.size(), 1.0);
                    ones_plus += out_neurons_j_vector;
                    delta = simple_product(delta, ones_plus);
                }else if(this->activation_function_name == "relu"){
                    //New delta: delta * [ vector de 0s y 1s ]  [] -> derivada de la función de activación
                    bnu::vector<double> mask(out_neurons_j_vector.size(), 0.0);
                    for(int i = 0; i < out_neurons_j_vector.size(); i++)
                        if(out_neurons_j_vector(i) > 0) mask(i) = 1.0;
                    delta = simple_product(delta, mask);
                }else{//Sigmoid por defecto
                    //New delta: delta * [ (Sj) * (1 - Sj) ]  [] -> derivada de la función de activación
                    delta = simple_product(delta, out_neurons_j_vector);
                    bnu::vector<double> ones(out_neurons_j_vector.size(), 1.0);
                    ones -= out_neurons_j_vector;
                    delta = simple_product(delta, ones);
                }
                
                if(debug) std::cout << "\tdelta_new = " << delta << "\n";
                //Actualizo el nuevo delta
                deltas[i_data] = delta;

                //Extrae Si, salida de la capa de entrada. Si es la primera matriz, sería el dato en sí. 1 al inicio por el bias.
                bnu::matrix<double> out_neurons_i(this->Ws[i_matrix].size1(), 1);
                if(i_matrix == 0){ //first matrix
                    bnu::matrix_row<bnu::matrix<double>> data_i_data(this->x_train, i_data);
                    out_neurons_i <<= 1, data_i_data;
                }else{
                    bnu::matrix_row<bnu::matrix<double>> out_neurons_i_data(this->out_neurons[i_matrix - 1], i_data);
                    out_neurons_i <<= 1, out_neurons_i_data;
                }
                if(debug) std::cout << "\tout_neurons_" << i_matrix << " = " << out_neurons_i << "\n";
                
                //Cálculo de la derivada
                bnu::matrix<double> derivadas_i = bnu::prod(out_neurons_i, delta);
                if(debug) std::cout << "\tderivadas_" << i_matrix << " = " << derivadas_i << "\n";
                //Acumulo la derivada
                Ws_derivades[i_matrix] += derivadas_i;
            }
        }
        if(debug) std::cout << "Ws_derivades[" << i_matrix << "] = " << Ws_derivades[i_matrix] << "\n";
    }
    //Actualizo Ws
    if(debug){
        std::cout << "\nPrev Ws:\n";
        for(int i = 0; i < this->Ws_size; i++)
            std::cout << "\tW[" << i << "] = " << this->Ws[i] << "\n";
        std::cout << "Ws_derivades:\n";
        for(int i = 0; i < Ws_derivades.size(); i++)
            std::cout << "\tW_deriv[" << i << "] = " << Ws_derivades[i] << "\n";
    }
    
    for(int i = 0; i < this->Ws.size(); i++)
        this->Ws[i] -= this->alpha * Ws_derivades[i];
    
    if(debug){
        std::cout << "Post Ws:\n";
        for(int i = 0; i < this->Ws_size; i++)
            std::cout << "\tW[" << i << "] = " << this->Ws[i] << "\n";
    }
    if(debug) std::cout << "\n----------Backward propagation End----------\n";
}

MLP::MLP(bnu::vector<int> &neuronas_by_layer){
    this->neuronas_by_layer = neuronas_by_layer;
    this->layers = neuronas_by_layer.size();
    this->Ws_size = layers + 1;
    this->Ws.resize(this->Ws_size);
    this->out_neurons.resize(this->layers);
}

MLP::~MLP(){
    //dtor
}

void MLP::train(bnu::matrix<double> &x_train, bnu::matrix<double> &y_train, bnu::matrix<double> &x_validation, bnu::matrix<double> &y_validation, int epochs, double alpha, std::string activation_function_name = "sigmoid",bool debug = false){
    //TODO: Función para resetear variables privadas
    this->x_train = x_train;
    this->y_train = y_train;
    this->x_validation = x_validation;
    this->y_validation = y_validation;
    this->alpha = alpha;
    this->activation_function_name = activation_function_name;
    this->n = this->x_train.size1();
    this->m = this->x_train.size2();
    this->classes = this->y_train.size2();
    this->out_o.resize(this->n, this->classes);
    this->out_o_softmax.resize(this->n, this->classes);
    
    this->Ws_rows.push_back(this->m + 1);
    for(int i = 0; i < this->layers; i++){
        this->Ws_rows.push_back(this->neuronas_by_layer[i] + 1);
        this->Ws_columns.push_back(this->neuronas_by_layer[i]);
    }
    this->Ws_columns.push_back(this->classes);

    for(int i = 0; i < this->Ws_size; i++)
        this->Ws[i].resize(this->Ws_rows[i], this->Ws_columns[i]);
        
    for(auto &W : Ws)
        this->fill_w(W);
   
    for(int i = 0; i < this->layers; i++)
        this->out_neurons[i].resize(this->n, this->neuronas_by_layer[i]);
    
    if(debug){
        std::cout << "x_train:" << this->x_train << "\n";
        std::cout << "y_train:" << this->y_train << "\n";
        for(int i=0; i<this->neuronas_by_layer.size(); i++)
            std::cout << "layer " << i + 1 << " with " << this->neuronas_by_layer[i] << " neurons\n";
        std::cout << this->Ws_size << " matrices of weights\n";
        for(int i=0; i < this->Ws_size; i++)
            std::cout << "Matrix " << i << ": " << this->Ws[i] << "\n";
        std::cout << "\n";
    }

    for(int epoch = 1; epoch <= epochs; epoch++){
        if(debug) std::cout << "\n*************************Epoch " << epoch << "*************************\n";
        this->forward_propagation(true, debug);
        this->error(debug);
        this->backward_propagation(debug);
    }
    if(debug) std::cout << "out_o_softmax = " << this->out_o_softmax << "\n";
    if(debug) std::cout << "y_train = " << this->y_train << "\n";
    if(debug) std::cout << "*************************Fin de entrenamiento*************************\n";
}

void MLP::error(bool debug = false){
    if(debug) std::cout << "\n------------Error---------------\n";
    bnu::matrix<double> error_matrix_training = this->out_o_softmax - this->y_train;
    bnu::matrix<double> error_matrix_validation = this->out_o_softmax - this->y_validation;
    if(debug){
        std::cout << "out_o = " << this->out_o << "\n";
        std::cout << "out_o_softmax = " << this->out_o_softmax << "\n";
        std::cout << "y_train = " << this->y_train << "\n";
        std::cout << "y_validation = " << this->y_validation << "\n";
        std::cout << "error_matrix_training = " << error_matrix_training << "\n";
        std::cout << "error_matrix_validation = " << error_matrix_validation << "\n";
    }
    //boost::range::transform(error_matrix, error_matrix.begin1(), square);
    for(int i = 0; i < error_matrix_training.size1(); i++){
        for(int j = 0; j < error_matrix_training.size2(); j++){
            error_matrix_training(i, j) = error_matrix_training(i, j) * error_matrix_training(i, j);
            error_matrix_validation(i, j) = error_matrix_validation(i, j) * error_matrix_validation(i, j);
        }
    }

    if(debug) std::cout << "square error training = " << error_matrix_training << "\n";
    if(debug) std::cout << "square error validation = " << error_matrix_validation << "\n";
    error_matrix_training = error_matrix_training / 2.0;
    error_matrix_validation = error_matrix_validation / 2.0;
    if(debug) std::cout << "square / 2.0 error training = " << error_matrix_training << "\n";
    if(debug) std::cout << "square / 2.0 error validation = " << error_matrix_validation << "\n";
    double error_training = bnu::sum(bnu::prod(bnu::scalar_vector<double>(error_matrix_training.size1()), error_matrix_training));
    double error_validation = bnu::sum(bnu::prod(bnu::scalar_vector<double>(error_matrix_validation.size1()), error_matrix_validation));
    this->errors_training.push_back(error_training);
    this->errors_validation.push_back(error_validation);
    if(debug) std::cout << "Error training: " << error_training << "\n";
    if(debug) std::cout << "Error validation: " << error_validation << "\n";
    if(debug) std::cout << "\n----------Error End-------------\n";
}

void MLP::predict(bnu::matrix<double> &x_test, bnu::matrix<double> &y_test){
    output_model_softmax_matrix.resize(x_test.size1(), this->classes);
    for(int i_test = 0; i_test < x_test.size1(); i_test++){
        bnu::matrix_row<bnu::matrix<double>> x_test_row(x_test, i_test);
        bnu::matrix_row<bnu::matrix<double>> y_test_row(y_test, i_test);
        bnu::vector<double> x_test_vector(x_test_row);
        this->predict_one(x_test_vector);
        
        //Escribe el output en la matriz
        for(int i = 0; i < this->classes; i++)
            this->output_model_softmax_matrix(i_test, i) = this->output_model_softmax(i);
    }
}

void MLP::predict_one(bnu::vector<double> &input_to_predict){
    this->input_model = input_to_predict;
    this->forward_propagation(false, false);
    return;
}   

void MLP::write_errors(std::string filename){
    std::ofstream file;
    /*
    write errors_training
    write errors_validation
    write output_model_softmax_matrix
    */
}