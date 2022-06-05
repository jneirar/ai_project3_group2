#include "MLP.cpp"
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/value.h>
#include <string>
bnu::matrix<double> x_train, y_train, x_validation, y_validation, x_test, y_test;

std::string file_input, file_output;

void read(){
    std::ifstream *file=new std::ifstream("encodes/img_encodings_1.json");
    Json::Value actualJson;
    Json::Reader reader;
    
    reader.parse(*file,actualJson);
    bnu::matrix<int> a(2,2);

    //cout<<"Total json data:"<<endl<<actualJson<<endl;
    std::cout<<actualJson["Train"]["Classes"].size()<<std::endl;
    //cout<<actualJson["Train"]["Images"]<<endl;
    //cout<<actualJson["Validation"]["Classes"]<<endl;
    //cout<<actualJson["Validation"]["Images"]<<endl;
    std::vector<std::string> tipos = {"Train","Validation","Test"};
    std::vector<std::string> xy = {"Classes","Images"};

    std::vector<bnu::matrix<double>> matrixes;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            int tam_clases_t_fil = actualJson[tipos[i]][xy[j]].size();
            int tam_clases_t_col = actualJson[tipos[i]][xy[j]][0].size();
            bnu::matrix<double> matr(tam_clases_t_fil,tam_clases_t_col);
            for (int i1 = 0; i1 <  tam_clases_t_fil ;i1++)
            {
                for (int j1 = 0; j1 < tam_clases_t_col; j1++)
                {
                    matr(i1,j1) = (actualJson[tipos[i]][xy[j]][i1][j1]).asDouble();
                }
                
            }
            matrixes.push_back(matr);
        }
    }
    x_train = matrixes[0];
    y_train = matrixes[1];
    x_validation = matrixes[2];
    y_validation = matrixes[3];
    x_test = matrixes[4];
    y_test = matrixes[5];
}

int main(int argc, char *argv[]){
    if(argc < 4) {
        std::cout << "Too low arguments. Usage: ./main <input_file> <output_file> <layers> ";
        std::cout << "<neurons_in_layer(1)1> <neurons_in_layer(2)> ... <neurons_in_layer(layers)> ";
        std::cout << "<activation_function> <epoch> <alpha> <debug> <seed>\n";
        return 0;
    }
    file_input = argv[1];
    file_output = argv[2];
    int layers = atoi(argv[3]);
    int arguments = 4 + layers + 5;
    if(argc < arguments){
        std::cout << "Few arguments. Usage: ./main <input_file> <output_file> <layers> ";
        std::cout << "<neurons_in_layer(1)1> <neurons_in_layer(2)> ... <neurons_in_layer(layers)> ";
        std::cout << "<activation_function> <epoch> <alpha> <debug> <seed>\n";
        return 0;
    }
    if(argc > arguments){
        std::cout << "Many arguments. Usage: ./main <input_file> <output_file> <layers> ";
        std::cout << "<neurons_in_layer(1)1> <neurons_in_layer(2)> ... <neurons_in_layer(layers)> ";
        std::cout << "<activation_function> <epoch> <alpha> <debug> <seed>\n";
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
    int seed = atoi(argv[8+layers]);
    if(seed < 1)
        seed = 1;
    read();
    mlp.train(x_train, y_train, x_validation, y_validation, epoch, alpha, activation_function_name, seed, debug);
    mlp.predict(x_test, y_test);
    mlp.write_errors(file_output);
    return 0;
}