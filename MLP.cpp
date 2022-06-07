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

void print_matrix(bnu::matrix<double> &m)
{
    for (int i = 0; i < m.size1(); i++)
    {
        for (int j = 0; j < m.size2(); j++)
        {
            //set precision of double
            std::cout << std::setw(15) << std::setprecision(8) << std::fixed << m(i, j) << " ";
        }
        std::cout << std::endl;
    }
}


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
    int limit_debug = 3;
    int n, m, layers, classes, Ws_size, layer_output;
    bnu::vector<int> neuronas_by_layer;
    std::vector<bnu::matrix<double>> Ws;
    std::vector<int> Ws_rows, Ws_columns;

    bnu::matrix<double> x_train, y_train, x_validation, y_validation;
    bnu::vector<double> input_model, output_model_softmax;
    bnu::matrix<double> output_model_softmax_validation;
    
    bnu::matrix<double> out_softmax_register; //for training
    std::vector<bnu::matrix<double>> out_neurons_register; //for training
    
    double alpha;
    std::string activation_function_name;
    std::vector<double> errors_training, errors_validation;

    void equal_m_v(bnu::matrix<double> &m, bnu::vector<double> &v){
        for(int i = 0; i < v.size(); i++)
            m(0, i) = v(i);
    }
    void prod_m_v_by_v(bnu::matrix<double> &m, bnu::vector<double> &v1, bnu::vector<double> &v2){
        for(int i = 0; i < v1.size(); i++)
            m(0, i) = v1(i) * v2(i);
    }

    void prod_m_v(bnu::matrix<double> &m, bnu::vector<double> &v){
        for(int i = 0; i < v.size(); i++)
            m(0, i) *= v(i);
    }
    void one_minus_v(bnu::vector<double> &vr, bnu::vector<double> &v){
        for(int i = 0; i < v.size(); i++)
            vr(i) = 1.0 - v(i);
    }
    void one_plus_v(bnu::vector<double> &vr, bnu::vector<double> &v){
        for(int i = 0; i < v.size(); i++)
            vr(i) = 1.0 + v(i);
    }
    void one_zero_v(bnu::vector<double> &vr, bnu::vector<double> &v){
        for(int i = 0; i < v.size(); i++){
            if(v(i) > 0) vr(i) = 1.0;
            else vr(i) = 0.0;
        }
    }
    
    void forward_propagation(bool for_training, int i_training, bool debug);
    void backward_propagation(bool debug);
    void error(bool debug);
    
public:
    MLP(bnu::vector<int> &neuronas_by_layer);
    ~MLP();
    void train(bnu::matrix<double> &x_train, bnu::matrix<double> &y_train, bnu::matrix<double> &x_validation, bnu::matrix<double> &y_validation, int epochs, double alpha, std::string activation_function_name, int seed, bool debug);
    void predict(bnu::matrix<double> &x_test, bnu::matrix<double> &y_test, bool debug);
    void write_errors(std::string filename);
};

void MLP::forward_propagation(bool for_training, int i_training = 0, bool debug = false){    
    //Forward para input_model
    bnu::vector<double> out_layer(this->input_model);
    
    //Para cada matriz, iniciamos desde input y terminamos con un output
    for(int i_matrix = 0; i_matrix < this->Ws_size; i_matrix++){
        //Define el input de la matriz (1 al inicio por el bias)
        bnu::vector<double> input_layer(out_layer.size() + 1);
        input_layer <<= 1, out_layer;
        if(debug) std::cout << "For matrix " << i_matrix << ":\n";
        if(debug) std::cout << "\tinput = " << input_layer << "\n";
        if(debug) std::cout << "\tW = " << this->Ws[i_matrix] << "\n";
        
        //Producto input x matriz
        out_layer = bnu::prod(input_layer, Ws[i_matrix]);
        if(debug) std::cout << "\toutput = " << out_layer << "\n";
        
        //Aplicamos la función de activación
        if(this->activation_function_name == "sigmoid")
            boost::range::transform(out_layer, out_layer.begin(), sigmoid);
        else if(this->activation_function_name == "relu")
            boost::range::transform(out_layer, out_layer.begin(), relu);
        else if(this->activation_function_name == "tanh")
            boost::range::transform(out_layer, out_layer.begin(), tanhiperbolica);
        
        if(debug) std::cout << "\tf_act(out) = " << out_layer << "\n";

        //Si es entrenamiento, guardamos la salida de cada capa
        if(for_training){
            for(int j = 0; j < out_layer.size(); j++)
                this->out_neurons_register[i_matrix](i_training, j) = out_layer(j);
        }
    }
    //Aplicamos la función softmax en out_layer para obtener la salida del modelo
    this->output_model_softmax = out_layer;
    boost::range::transform(this->output_model_softmax, this->output_model_softmax.begin(), exponencial);
    double denominator = boost::accumulate(this->output_model_softmax, 0.0);
    this->output_model_softmax /= denominator;

    //Si es entrenamiento, guardamos esta salida en out_softmax_register
    if(for_training){
        for(int j = 0; j < this->output_model_softmax.size(); j++)
            this->out_softmax_register(i_training, j) = this->output_model_softmax(j);
    }
}

void MLP::backward_propagation(bool debug = false){
    if(debug) std::cout << "\n------------Backward propagation------------\n";
    //Cálculo de las derivadas respecto a W, se suma la derivada de cada dato
    std::vector<bnu::matrix<double>> Ws_derivades(Ws.size());
    for(int i = 0; i < Ws.size(); i++)
        Ws_derivades[i].resize(this->Ws[i].size1(), this->Ws[i].size2(), 0);
        
    
    std::vector<bnu::matrix<double>> deltas(n);
    bnu::matrix<double> delta;
    bnu::vector<double> delta_vector; //Temporal para el cálculo de delta
    
    for(int i_matrix = this->Ws_size - 1; i_matrix >= 0; i_matrix--){
        if(debug) std::cout << "In matrix " << i_matrix << ":\n";
        //bool local_debug = debug;
        bool local_debug = 0;
        for(int i_data = 0; i_data < this->n; i_data++){
            if(local_debug) std::cout << "- data " << i_data << ":\n";
            
            //Si es la última capa
            if(i_matrix == this->Ws_size - 1){
                bnu::vector<double> out_d_i_vector = bnu::matrix_row<bnu::matrix<double>> (this->y_train, i_data),
                                    out_o_i_vector = bnu::matrix_row<bnu::matrix<double>> (this->out_neurons_register[i_matrix], i_data),
                                    out_o_softmax_i_vector = bnu::matrix_row<bnu::matrix<double>> (this->out_softmax_register, i_data);
                //(S'o - Sd)
                delta_vector = out_o_softmax_i_vector - out_d_i_vector;
                delta.resize(1, delta_vector.size());
                this->equal_m_v(delta, delta_vector);
                if(this->activation_function_name == "sigmoid"){
                    //Derivada de la función de activación: (1 - So) * (So)
                    //(S'o - Sd) * [ (So) ]
                    this->prod_m_v(delta, out_o_i_vector);
                    //(S'o - Sd) * [ (So) * (1 - So) ]
                    this->one_minus_v(delta_vector, out_o_i_vector);
                    this->prod_m_v(delta, delta_vector);
                }else if(this->activation_function_name == "tanh"){
                    //Derivada de la función de activación: (1 - So) * (1 + So)
                    //(S'o - Sd) * [ (1 - So) ]
                    this->one_minus_v(delta_vector, out_o_i_vector);
                    this->prod_m_v(delta, delta_vector);
                    //(S'o - Sd) * [ (1 - So) * (1 + So) ]
                    this->one_plus_v(delta_vector, out_o_i_vector);
                    this->prod_m_v(delta, delta_vector);
                }else if(this->activation_function_name == "relu"){
                    //Derivada de la función de activación: 0 si So < 0, 1 si So > 0
                    this->one_zero_v(delta_vector, out_o_i_vector);
                    this->prod_m_v(delta, delta_vector);
                }
                
                //Obtengo una matriz delta (1 x classes)
                if(local_debug) std::cout << "\tdelta = " << delta << "\n";
                
                //Guardo el delta para la siguiente matriz
                deltas[i_data] = delta;
                
                //(Si) -> Salida de la última capa oculta (1 al inicio por el bias)
                bnu::matrix_row<bnu::matrix<double>> out_neurons_i_vector(this->out_neurons_register[this->layer_output - 1], i_data);
                bnu::matrix<double> out_neurons_i(out_neurons_i_vector.size() + 1, 1);
                out_neurons_i <<= 1, out_neurons_i_vector;
                if(local_debug) std::cout << "\tout_neurons_i = " << out_neurons_i << "\n";
                
                //Derivada = Si * delta, da una matriz del mismo tamaño que Wi
                bnu::matrix<double> derivadas_0 = bnu::prod(out_neurons_i, delta);
                if(local_debug) std::cout << "\tderivadas_ " << i_matrix << " = " << derivadas_0 << "\n";
                //Acumulo la derivada
                Ws_derivades[i_matrix] += derivadas_0;
            }else{
                //Extrae Ws de la sgte capa menos la primera fila (menos el biass)
                bnu::matrix<double> Ws_next(this->Ws[i_matrix + 1].size1() - 1, this->Ws[i_matrix + 1].size2());
                for(int i = 0; i < Ws_next.size1(); i++)
                    for(int j = 0; j < Ws_next.size2(); j++)
                        Ws_next(i, j) = this->Ws[i_matrix + 1](i + 1, j);
                if(local_debug) std::cout << "\tWs_next = " << Ws_next << "\n";
                
                //Extrae el prev delta correspondiente al dato i_data
                delta = deltas[i_data];
                if(local_debug) std::cout << "\tdelta = " << delta << "\n";
                //Sum de delta * w de la capa previa:
                delta = bnu::prod(Ws_next, bnu::trans(delta));
                delta = bnu::trans(delta);
                if(local_debug) std::cout << "\tdelta x w = " << delta << "\n";
                
                //Extrae Sj, salida de la capa de salida de la matriz
                bnu::vector<double> out_neurons_j_vector = bnu::matrix_row<bnu::matrix<double>> (this->out_neurons_register[i_matrix], i_data);
                if(local_debug) std::cout << "\tout_neurons_j = " << out_neurons_j_vector << "\n";

                delta_vector.resize(out_neurons_j_vector.size(), 0);
                if(this->activation_function_name == "sigmoid"){
                    //New delta: delta * [ (Sj) * (1 - Sj) ]
                    this->prod_m_v(delta, out_neurons_j_vector);
                    this->one_minus_v(delta_vector, out_neurons_j_vector);
                    this->prod_m_v(delta, delta_vector);
                }else if(this->activation_function_name == "tanh"){
                    //New delta: delta * [ (1 + Sj) * (1 - Sj) ] 
                    this->one_plus_v(delta_vector, out_neurons_j_vector);
                    this->prod_m_v(delta, delta_vector);
                    this->one_minus_v(delta_vector, out_neurons_j_vector);
                    this->prod_m_v(delta, delta_vector);
                }else if(this->activation_function_name == "relu"){
                    //New delta: delta * [ vector de 0s y 1s ] 
                    this->one_zero_v(delta_vector, out_neurons_j_vector);
                    this->prod_m_v(delta, delta_vector);
                }
                
                if(local_debug) std::cout << "\tdelta_new = " << delta << "\n";
                //Actualizo el nuevo delta
                deltas[i_data] = delta;

                //Extrae Si, salida de la capa de entrada. Si es la primera matriz, sería el dato en sí. 1 al inicio por el bias.
                bnu::matrix<double> out_neurons_i(this->Ws[i_matrix].size1(), 1);
                if(i_matrix == 0){ //first matrix
                    bnu::matrix_row<bnu::matrix<double>> data_i_data(this->x_train, i_data);
                    out_neurons_i <<= 1, data_i_data;
                }else{
                    bnu::matrix_row<bnu::matrix<double>> out_neurons_i_data(this->out_neurons_register[i_matrix - 1], i_data);
                    out_neurons_i <<= 1, out_neurons_i_data;
                }
                if(local_debug) std::cout << "\tout_neurons_" << i_matrix << " = " << out_neurons_i << "\n";
                
                //Cálculo de la derivada
                bnu::matrix<double> derivadas_i = bnu::prod(out_neurons_i, delta);
                if(local_debug) std::cout << "\tderivadas_" << i_matrix << " = " << derivadas_i << "\n";
                //Acumulo la derivada
                Ws_derivades[i_matrix] += derivadas_i;
            }
            if(i_data == this->limit_debug) local_debug = 0;
        }
        if(debug) std::cout << "Ws_derivades[" << i_matrix << "] = " << Ws_derivades[i_matrix] << "\n";
    }
    //Actualizo Ws
    if(debug){
        std::cout << "\nPrev Ws:\n";
        for(int i = 0; i < this->Ws_size; i++){
            std::cout << "\tW[" << i << "] = " << this->Ws[i] << "\n";
            print_matrix(this->Ws[i]); 
            std::cout << "\n";
        }
        std::cout << "Ws_derivades:\n";
        for(int i = 0; i < Ws_derivades.size(); i++){
            std::cout << "\tW_deriv[" << i << "] = " << Ws_derivades[i] << "\n";
            print_matrix(Ws_derivades[i]);
            std::cout << "\n";
        }
    }
    
    for(int i = 0; i < this->Ws.size(); i++)
        this->Ws[i] -= this->alpha * Ws_derivades[i];
    
    if(debug){
        std::cout << "Post Ws:\n";
        for(int i = 0; i < this->Ws_size; i++){
            std::cout << "\tW[" << i << "] = " << this->Ws[i] << "\n";
            print_matrix(this->Ws[i]);
            std::cout << "\n";
        }
    }
    if(debug) std::cout << "\n----------Backward propagation End----------\n";
}

MLP::MLP(bnu::vector<int> &neuronas_by_layer){
    this->neuronas_by_layer = neuronas_by_layer;
    this->layers = neuronas_by_layer.size();
    this->Ws_size = layers + 1;
    this->Ws.resize(this->Ws_size);
    this->out_neurons_register.resize(this->layers + 1);
    this->layer_output = this->layers;
}

MLP::~MLP(){
    //dtor
}

void MLP::train(bnu::matrix<double> &x_train, bnu::matrix<double> &y_train, bnu::matrix<double> &x_validation, bnu::matrix<double> &y_validation, int epochs, double alpha, std::string activation_function_name = "sigmoid", int seed = 0, bool debug = false){
    this->x_train = x_train;
    this->y_train = y_train;
    this->x_validation = x_validation;
    this->y_validation = y_validation;
    this->alpha = alpha;
    this->activation_function_name = activation_function_name;
    this->n = this->x_train.size1();
    this->m = this->x_train.size2();
    this->classes = this->y_train.size2();

    this->input_model.resize(this->m);
    this->output_model_softmax.resize(this->classes);
    this->out_softmax_register.resize(this->n, this->classes);

    this->output_model_softmax_validation.resize(x_validation.size1(), this->classes);
    
    //Creación de las matrices W
    this->Ws_rows.push_back(this->m + 1);
    for(int i = 0; i < this->layers; i++){
        this->Ws_rows.push_back(this->neuronas_by_layer[i] + 1);
        this->Ws_columns.push_back(this->neuronas_by_layer[i]);
    }
    this->Ws_columns.push_back(this->classes);

    for(int i = 0; i < this->Ws_size; i++)
        this->Ws[i].resize(this->Ws_rows[i], this->Ws_columns[i]);

    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(-0.5, 0.5);
    
    for(auto &W : Ws){
        for (int i = 0; i < W.size1(); i++)
            for (int j = 0; j < W.size2(); j++)
                W(i, j) = distribution(generator);
    }
   
    for(int i = 0; i < this->layers; i++)
        this->out_neurons_register[i].resize(this->n, this->neuronas_by_layer[i]);
    this->out_neurons_register[this->layer_output].resize(this->n, this->classes);
    
    if(debug){
        std::cout << "x_train: [" << this->x_train.size1() << "." << this->x_train.size2() << "]\n";
        std::cout << "y_train: [" << this->y_train.size1() << "," << this->y_train.size2() << "]\n";
        std::cout << "x_validation: [" << this->x_validation.size1() << "," << this->x_validation.size2() << "]\n";
        std::cout << "y_validation: [" << this->y_validation.size1() << "," << this->y_validation.size2() << "]\n";
        for(int i=0; i<this->neuronas_by_layer.size(); i++)
            std::cout << "layer " << i + 1 << " with " << this->neuronas_by_layer[i] << " neurons\n";
        std::cout << this->Ws_size << " matrices of weights\n";
        for(int i=0; i < this->Ws_size; i++)
            std::cout << "Matrix " << i << ": " << this->Ws[i] << "\n";
        std::cout << "\n";
    }

    for(int epoch = 1; epoch <= epochs; epoch++){
        if(debug) std::cout << "\n*************************Epoch " << epoch << "*************************\n";
        if(debug) std::cout << "\n-------------Forward propagation------------\n";
        //bool local_debug = debug;
        bool local_debug = 0;
        for(int i_data = 0; i_data < this->n; i_data++){
            this->input_model = bnu::matrix_row<bnu::matrix<double>> (this->x_train, i_data);
            if(local_debug) std::cout << "Input model: " << this->input_model << "\n";
            this->forward_propagation(true, i_data, local_debug);
            if(i_data == this->limit_debug) local_debug = 0;
        }
        if(debug){
            std::cout << "Salidas de las neuronas por cada dato: \n";
            for(int i = 0; i < this->out_neurons_register.size(); i++){
                std::cout << "\nMatrix " << i << ":\n";
                print_matrix(this->out_neurons_register[i]);
            }
            std::cout << "\nSalidas softmax del modelo: \n";
            print_matrix(this->out_softmax_register);
        } 
        if(debug) std::cout << "\n-----------Forward propagation End----------\n";
        //if(debug) std::cout << "\n-------Forward propagation For Validation-------\n";
        //local_debug = debug;
        local_debug = 0;
        for(int i_data = 0; i_data < this->x_validation.size1(); i_data++){            
            this->input_model = bnu::matrix_row<bnu::matrix<double>> (this->x_validation, i_data);
            if(local_debug) std::cout << "Input model: " << this->input_model << "\n";
            this->forward_propagation(false, i_data, local_debug);
            //Guardar la salida del modelo en output_model_softmax_validation
            for(int j = 0; j < this->output_model_softmax.size(); j++)
                this->output_model_softmax_validation(i_data, j) = this->output_model_softmax(j);
            if(i_data == this->limit_debug) local_debug = 0;
        }
        //if(debug) std::cout << "\n-----Forward propagation End For Validation-----\n";
        if(debug) std::cout << "\n------------Error---------------\n";
        this->error(debug);
        if(debug) std::cout << "\n----------Error End-------------\n";
        this->backward_propagation(debug);
    }
    if(debug) std::cout << "out_train = " << this->out_softmax_register << "\n";
    if(debug) std::cout << "y_train = " << this->y_train << "\n";
    if(debug) std::cout << "out_validation = " << this->output_model_softmax_validation << "\n";
    if(debug) std::cout << "y_validation = " << this->y_validation << "\n";
    if(debug) std::cout << "*************************Fin de entrenamiento*************************\n\n";
}

void MLP::error(bool debug = false){
    if(debug){
        std::cout << "out training = " << this->out_softmax_register << "\n";
        std::cout << "y_train = " << this->y_train << "\n";
        std::cout << "out validation = " << this->output_model_softmax_validation << "\n";
        std::cout << "y_validation = " << this->y_validation << "\n";
    }
    double error_training = 0.0, error_validation = 0.0;
    for(int i = 0; i < this->y_train.size1(); i++)
        for(int j = 0; j < this->y_train.size2(); j++)
            error_training -= (this->y_train(i, j) * log(this->out_softmax_register(i, j)));
            
    for(int i = 0; i < this->y_validation.size1(); i++)
        for(int j = 0; j < this->y_validation.size2(); j++)
            error_validation -= (this->y_validation(i, j) * log(this->output_model_softmax_validation(i, j)));    
    error_training /= this->y_train.size1();
    error_validation /= this->y_validation.size1();

    this->errors_training.push_back(error_training);
    this->errors_validation.push_back(error_validation);
    if(debug) std::cout << "Error training: " << error_training << "\n";
    if(debug) std::cout << "Error validation: " << error_validation << "\n";
}

void MLP::predict(bnu::matrix<double> &x_test, bnu::matrix<double> &y_test, bool debug = false){
    this->output_model_softmax_validation.resize(x_test.size1(), this->classes, 0);
    this->y_validation = y_test;
    if(debug) std::cout << "*************************Prediction*************************\n";
    bool debug_local = debug;
    for(int i_test = 0; i_test < x_test.size1(); i_test++){
        bnu::matrix_row<bnu::matrix<double>> x_row(this->x_train, i_test);
        this->input_model = x_row;
        if(debug_local) std::cout << "Input model: " << this->input_model << "\n";
        this->forward_propagation(false, i_test, debug_local);
        //Guardar la salida del modelo en output_model_softmax_validation
        for(int j = 0; j < this->output_model_softmax.size(); j++)
            this->output_model_softmax_validation(i_test, j) = this->output_model_softmax(j);
        if(i_test == this->limit_debug) debug_local = 0;
    }
    if(debug) std::cout << "output test: " << this->output_model_softmax_validation << "\n";
    if(debug) std::cout << "***********************Prediction End***********************\n";
}

void MLP::write_errors(std::string filename){
    std::ofstream file;
    file.open(filename + "_" + this->activation_function_name + ".txt", std::ostream::trunc);
    for(int i = 0; i < this->errors_training.size(); i++)
        file << this->errors_training[i] << " " << this->errors_validation[i] << "\n";
    int aciertos = 0;
    file << "\n";
    file << "Salidas test: \n";
    double val1, val2;
    int i1, i2;
    for(int i = 0; i < this->output_model_softmax_validation.size1(); i++){
        i1 = 0;
        val1 = y_validation(i, 0);
        for(int j = 0; j < this->classes; j++){
            file << this->y_validation(i, j) << " ";
            if(this->y_validation(i, j) > val1){
                val1 = this->y_validation(i, j);
                i1 = j;
            }
        }
        file << "\n";
        i2 = 0;
        val2 = this->output_model_softmax_validation(i, 0);
        for(int j = 0; j < this->output_model_softmax_validation.size2(); j++){
            file << this->output_model_softmax_validation(i, j) << " ";
            if(this->output_model_softmax_validation(i, j) > val2){
                val2 = this->output_model_softmax_validation(i, j);
                i2 = j;
            }
        }
        file << "\n";
        if(i1 == i2) aciertos++;
    }
    file << "\nAciertos: " << aciertos << "/" << this->output_model_softmax_validation.size1() << " = " << (double)aciertos/this->output_model_softmax_validation.size1()*100 << "%\n";
    std::cout << "\nAciertos test: " << (double)aciertos/this->output_model_softmax_validation.size1()*100 << "%\n";
    file << "\n";

    aciertos = 0;
    file << "Salidas training: \n";
    for(int i = 0; i < this->n; i++){
        i1 = 0;
        i2 = 0;
        val1 = y_train(i, 0);
        for(int j = 0; j < this->classes; j++){
            file << this->y_train(i, j) << " ";
            if(this->y_train(i, j) > val1){
                val1 = this->y_train(i, j);
                i1 = j;
            }
        }
        file << "\n";
        val2 = this->out_softmax_register(i, 0);
        for(int j = 0; j < this->classes; j++){
            file << this->out_softmax_register(i, j) << " ";
            if(this->out_softmax_register(i, j) > val2){
                val2 = this->out_softmax_register(i, j);
                i2 = j;
            }
        }
        if(i1 == i2) aciertos++;
        file << "\n";
    }file << "\n";
    file << "\nAciertos: " << aciertos << "/" << this->n << " = " << (double)aciertos/this->n*100 << "%\n\n";
    std::cout << "\nAciertos training: " << (double)aciertos/this->n*100 << "%\n";
    
    file.close();
    std::cout << this->activation_function_name << " finish\n";
    /*
    write errors_training
    write errors_validation
    write output_model_softmax_matrix
    */
}