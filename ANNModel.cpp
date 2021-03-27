#include <time.h>
#include <iostream>
#include "matrix.cpp"
#include "activation.cpp"
#include "dataset.cpp"
#include <functional>
#include <queue>
using namespace MatrixLib;
using namespace std;

// 简单的神经网络模型
class ANN {
    static const int layer = 3;                                 // 网络层数
    static const int training_amount = 60000;                   // 训练集规模
    static const int validation_start  = 50000;                 // 验证集开始位置(在训练集中)
    static const int test_amount  = 10000;                      // 测试集规模
    static constexpr int dimension[layer] = {14*14, 64, 10};    // 层级神经元数量分配
    const string model_name = "model_3_9631.nn";                // 模型名称, 用于装载与保存
    static const int valid_layer_start = 0;                     // 网络有效开始层(针对后向传播)

    // 层级激活函数
    const vector<Matrix(*)(const Matrix&)> activation =   {linear, tanh, softmax};
    const vector<Matrix(*)(const Matrix&)> d_activation = {d_linear, d_tanh, d_softmax};

    // 层级参数随机范围初始化列表
    const vector<vector<vector<float>>> initList = {
        {
            {0, 0}, // b0
            {1, 1} // w0
        },
        {
            {0, 0},// b1
            // w1
            {-(float)sqrt(6.f/(dimension[0]+dimension[1])), +(float)sqrt(6.f/(dimension[0]+dimension[1]))}
        },
        {
            {0, 0},// b2
            // w2
            {-(float)sqrt(6.f/(dimension[1]+dimension[2])), +(float)sqrt(6.f/(dimension[1]+dimension[2]))}
        }
    };
    
    /* 前向传播结果参数 */
    struct ProcessParams {      
        Matrix Z[layer];        // Z[i] = W[i]*A[i-1] + B[i],  每一层的输入处理层
        Matrix A[layer];        // A[i] = Activation[i](Z[i]), 每一层的输出层
    };

    /* 后向传播结果参数 */
    struct Gradients {
        Matrix dW[layer];       // 梯度dW[i]
        Matrix dB[layer];       // 梯度dB[i]
        float loss;             // 损失量
    };

    Dataset dataset;            // 数据集连接处理对象实例化

public:

    ANN() : dataset(training_amount, test_amount) {}

    void init_b(vector<Matrix>& B);     // B参数初始化
    void init_w(vector<Matrix>& W);     // W参数初始化

    // 前向传播, 只作为预测功能
    Matrix forwardPropagation_predict(Matrix& X, vector<Matrix>& W, vector<Matrix>& B);
    // 前向传播, 返回过程参数，用于后向传播求梯度
    ProcessParams forwardPropagation(Matrix& X, vector<Matrix>& W, vector<Matrix>& B);
    // 后向传播, 返回各层级梯度参数以及损失量
    Gradients backPropagation(Matrix& X, vector<Matrix>& W, vector<Matrix>& B, int rightLabel, ProcessParams& procParams);

    // 批量训练
    float train_batch(vector<Matrix>& W, vector<Matrix>& B, 
                      double learningRate, 
                      vector<int>& startList,
                      int start, int batch_size
                      );

    // 计算目标损失量
    float get_loss(Matrix& predictAL, int rightLabel);

    // <loss, accuracy> 测试 验证 训练集 的损失量以及准确率
    pair<float, float> test_lossAndAccuracy(vector<Matrix>& W, vector<Matrix>& B);
    pair<float, float> validation_lossAndAccuracy(vector<Matrix>& W, vector<Matrix>& B);
    pair<float, float> training_lossAndAccuracy(vector<Matrix>& W, vector<Matrix>& B);
    // 测试每种标签的准确率以及损失量
    void eachNumTestTesting();

    bool save(vector<Matrix>& W, vector<Matrix>& B);        // 保存模型
    bool load(vector<Matrix>& W, vector<Matrix>& B);        // 装载模型

    // 训练主入口
    void main() {
        dataset.init();

        randInit();
        vector<Matrix> W, B;
        init_w(W);
        init_b(B);
        load(W, B);
        validation_lossAndAccuracy(W, B);
        // training_lossAndAccuracy(W, B);

        int epoch = 2;
        double learning_rate = 0.5;
        int batch_size = 10;
        while (epoch-- > 0) {
            // this epoch
            
            double sum_loss = 0;
            queue<float> lossWindow;

            vector<int> startList(validation_start);
            for (int j = 0; j < validation_start; ++j) startList[j] = j;
            random_shuffle(startList.begin(), startList.end());

            for (int i = 0; i < validation_start/batch_size; ++i) {
                // shuffle 每次输入顺序
                float loss = train_batch(W, B, learning_rate, startList, i*batch_size, batch_size);
                sum_loss += loss;
                lossWindow.push(loss);
                if (lossWindow.size() > 100) {
                    sum_loss -= lossWindow.front();
                    lossWindow.pop();
                }

                // cout << i << " : " << loss << endl;
                if (i && i % 100 == 0) cout << i  << " accu loss = " << sum_loss/min(1+i, 100) << " , loss = " << loss << endl;
            }

            cout << "this trained data:" << endl;
            auto&& [loss, accu] = validation_lossAndAccuracy(W, B);
            test_lossAndAccuracy(W, B);
            char c;
            cin >> c;
            if (c == 'y')
                save(W, B);

            // save(W, B);
        }
    }   


};

int main() {
    ANN ann;
    ann.main();

    system("pause");
}



void ANN::init_b(vector<Matrix>& B) {
    B.resize(layer);
    for (int i = 0; i < layer; ++i) {
        B[i] = Matrix(1, dimension[i], initList[i][0][0], initList[i][0][1]);
    }
}
void ANN::init_w(vector<Matrix>& W) {
    W.resize(layer);
    W[0] = Matrix(1, dimension[0], initList[0][1][0], initList[0][1][1]);

    for (int i = 1; i < layer; ++i) {
        W[i] = Matrix(dimension[i], dimension[i-1], initList[i][1][0], initList[i][1][1]);
    }
}

Matrix ANN::forwardPropagation_predict(Matrix& X, vector<Matrix>& W, vector<Matrix>& B) {
    // 需要保存的数据 Z, A
    // Z[i] 为 第i层的输入数据，Zi = W[i]*A[i-1] + B[i]
    // A[i] 为 第i层的被激活后最后输出的数据，A[i] = F(Z[i]);

    ProcessParams procParams;

    // 第0层只需要进行对生数据进行对W[0]点积加偏置B[0]即可
    procParams.Z[0] = X.dotElem(W[0]) + B[0]; 
    procParams.A[0] = activation[0](procParams.Z[0]);       

    for (int i = 1; i < layer; ++i) {
        // 将A[i-1]的数据都和W[i]的每一行（即每一个神经单元）进行点积，结果每一行为一个数字
        // 最后只会保留该层神经单元的个数行，为了后序方便统一处理，需要转置为一行
        procParams.Z[i] = W[i].dotRow(procParams.A[i-1]) + B[i]; 
        procParams.Z[i].transpose();
        procParams.A[i] = activation[i](procParams.Z[i]);
    }
    return procParams.A[layer-1];
}

ANN::ProcessParams ANN::forwardPropagation(Matrix& X, vector<Matrix>& W, vector<Matrix>& B) {
    // 需要保存的数据 Z, A
    // Z[i] 为 第i层的输入数据，Zi = W[i]*A[i-1] + B[i]
    // A[i] 为 第i层的被激活后最后输出的数据，A[i] = F(Z[i]);

    ProcessParams procParams;

    // 第0层只需要进行对生数据进行对W[0]点积加偏置B[0]即可
    procParams.Z[0] = X.dotElem(W[0]) + B[0]; 
    procParams.A[0] = activation[0](procParams.Z[0]);       

    for (int i = 1; i < layer; ++i) {
        // 将A[i-1]的数据都和W[i]的每一行（即每一个神经单元）进行点积，结果每一行为一个数字
        // 最后只会保留该层神经单元的个数行，为了后序方便统一处理，需要转置为一行
        procParams.Z[i] = W[i].dotRow(procParams.A[i-1]) + B[i]; 
        procParams.Z[i].transpose();
        procParams.A[i] = activation[i](procParams.Z[i]);
    }
    return procParams;
}

ANN::Gradients ANN::backPropagation(Matrix& X, vector<Matrix>& W, vector<Matrix>& B, int rightLabel, ProcessParams& procParams) {
    // 正确标签向量
    Matrix Y(1, dimension[layer-1], 0);
    Y[0][rightLabel] = 1;

    Gradients G;
    Matrix dZ[layer];

    // 策略：从最后一层开始往前求dZ，顺便求dB与dW

    // 初始化最后一层
    dZ[layer-1] = (procParams.A[layer-1] - Y).dotElem(d_activation[layer-1](procParams.Z[layer-1]));
    G.dB[layer-1] = dZ[layer-1];
    for (int i = 0; i < dimension[layer-1]; ++i) {
        auto temp = unfold(procParams.A[layer-2] * dZ[layer-1][0][i]);
        G.dW[layer-1].addRow(temp);
    }

    for (int L = layer-2; L >= valid_layer_start; --L) {




        Matrix dZL_dZLPSum(1, dimension[L]);
        for (int i = 0; i < dimension[L]; ++i) {
            dZL_dZLPSum[0][i] = dZ[L+1].dotALL(Matrix(W[L+1].getCols(i)));
        }
        
        dZ[L] = d_activation[L](procParams.Z[L]).dotElem(dZL_dZLPSum);
        G.dB[L] = dZ[L];

        if (L > valid_layer_start) {
            for (int i = 0; i < dimension[L]; ++i) {
                auto temp = unfold(procParams.A[L-1] * dZ[L][0][i]);
                G.dW[L].addRow(temp);
            }
        }
    }

    G.loss = (double)0.5 * (procParams.A[layer-1] - Y).norm2();
    return G;
}

float ANN::train_batch(vector<Matrix>& W, vector<Matrix>& B, 
                    double learningRate, 
                    vector<int>& startList,
                    int start, int batch_size
                    ) {
    // int start = rand() % (validation_start - batchSize);
    int label;
    auto& X = dataset.readTrainingImg(label, startList[start]);

    auto procParams = forwardPropagation(X, W, B);
    Gradients G = backPropagation(X, W, B, label, procParams);

    for (int i = start+1; i < start+batch_size; ++i) {
        X = dataset.readTrainingImg(label, startList[i]);
        auto procParams = forwardPropagation(X, W, B);
        Gradients G_temp = backPropagation(X, W, B, label, procParams);

        for (int j = valid_layer_start; j < layer; ++j) {
            G.dB[j] += G_temp.dB[j];
            G.dW[j] += G_temp.dW[j];
        }
        G.loss += G_temp.loss;
    }

    for (int j = valid_layer_start; j < layer; ++j) {
        G.dB[j] *= learningRate / (double)batch_size;
        B[j] -= G.dB[j];
        G.dW[j] *= learningRate / (double)batch_size;
        W[j] -= G.dW[j];
    }
    G.loss /= batch_size;
    return G.loss;
}

float ANN::get_loss(Matrix& predictAL, int rightLabel) {
    // 正确标签向量
    Matrix Y(1, dimension[layer-1], 0);
    Y[0][rightLabel] = 1;

    return (double)0.5 * (predictAL - Y).norm2();
}

// <loss, accuracy>
pair<float, float> ANN::test_lossAndAccuracy(vector<Matrix>& W, vector<Matrix>& B) {
    int count_right = 0;
    double accu_loss = 0;

    for (int i = 0; i < test_amount; ++i) {
        int rightLabel;
        auto& X = dataset.readTestImg(rightLabel, i);
        auto predictAL = forwardPropagation_predict(X, W, B);

        auto predictLabel = predictAL.maxer().second;
        count_right += rightLabel == predictLabel;

        accu_loss += get_loss(predictAL, rightLabel);
    }

    float accuracy = (count_right / (float)test_amount);
    cout << "[test]accuracy   = " << accuracy << endl;
    cout << "[test]accum loss = " << accu_loss << endl;

    return {accu_loss, (count_right / (float)test_amount)};
}
pair<float, float> ANN::validation_lossAndAccuracy(vector<Matrix>& W, vector<Matrix>& B) {
    int count_right = 0;
    double accu_loss = 0;

    for (int i = validation_start; i < training_amount; ++i) {
        int rightLabel;
        auto& X = dataset.readTrainingImg(rightLabel, i);
        auto predictAL = forwardPropagation_predict(X, W, B);

        auto predictLabel = predictAL.maxer().second;
        count_right += rightLabel == predictLabel;

        accu_loss += get_loss(predictAL, rightLabel);
    }

    float accuarcy = (count_right / (float)(training_amount - validation_start));
    cout << "[validation]accuracy   = " << accuarcy << endl;
    cout << "[validation]accum loss = " << accu_loss << endl;
    return {accu_loss, (count_right / (float)test_amount)};
}
pair<float, float> ANN::training_lossAndAccuracy(vector<Matrix>& W, vector<Matrix>& B) {
    int count_right = 0;
    double accu_loss = 0;

    for (int i = 0; i < validation_start; ++i) {
        int rightLabel;
        auto& X = dataset.readTrainingImg(rightLabel, i);
        auto predictAL = forwardPropagation_predict(X, W, B);

        auto predictLabel = predictAL.maxer().second;
        count_right += rightLabel == predictLabel;

        accu_loss += get_loss(predictAL, rightLabel);
    }
    accu_loss /= 5.f;

    float accuarcy = count_right / (float)validation_start;
    cout << "[training]accuracy   = " << accuarcy << endl;
    cout << "[training]accum loss = " << accu_loss << endl;
    return {accu_loss, (count_right / (float)test_amount)};
}


bool ANN::save(vector<Matrix>& W, vector<Matrix>& B) {
    // W[0], W[1], .... W[L-1]
    // B[0], B[1], .... B[L-1]
    // each = Matrix

    // L=2
    // =W=
    // 
    // 
    // =B=
    // 

    ofstream ofs;
    ofs.open(model_name, ios::out);
    if (!ofs.is_open()) return false;

    ofs << to_string(W.size());
    ofs << '\n';
    ofs << "=W=\n";

    for (int i = 0; i < W.size(); ++i) {
        string temp = W[i].serialize();
        ofs << to_string(temp.size());
        ofs << '\n';
        ofs << temp;
        ofs << '\n';
    }
    
    ofs << "=B=\n";
    for (int i = 0; i < W.size(); ++i) {
        string temp = B[i].serialize();
        ofs << to_string(temp.size());
        ofs << '\n';
        ofs << temp;
        ofs << '\n';
    }

    ofs.flush();
    ofs.close();

    cout << "save ok!" << endl;
    return true;
}
bool ANN::load(vector<Matrix>& W, vector<Matrix>& B) {
    ifstream ifs;
    ifs.open(model_name, ios::in);
    
    if (!ifs.is_open()) {
        cout << "load fail!" << endl;
        return false;
    }
    
    // 处理第1行
    char indicate[32];
    ifs.getline(indicate, 32);
    if (stoi(indicate) == 0 || string(indicate) == "") {
        cout << "load fail!" << endl;
        return false;
    }


    // 处理W第1行
    ifs.getline(indicate, 32);

    for (int i = 0; i < W.size(); ++i) {
        // 处理第3行
        ifs.getline(indicate, 32);
        int amount = stoi(indicate) + 1;

        char serialized[amount];
        ifs.getline(serialized, amount);

        W[i] = Matrix(string(serialized));
    }
    // 处理B第1行
    ifs.getline(indicate, 32);
    for (int i = 0; i < B.size(); ++i) {
        ifs.getline(indicate, 32);
        int amount = stoi(indicate) + 1;

        char serialized[amount];
        ifs.getline(serialized, amount);

        B[i] = Matrix(string(serialized));
    }

    cout << "load success!" << endl;
    return true;
}


void ANN::eachNumTestTesting() {
    dataset.initTestDataset();
    randInit();
    vector<Matrix> W, B;
    init_w(W);
    init_b(B);
    load(W, B);
    test_lossAndAccuracy(W, B);

    for (int n = 0; n < 10; ++n) {
        int count = 0;
        int correct = 0;
        double loss = 0;
        for (int i = 0; i < 10000; ++i) {
            int label;
            auto X = dataset.readTestImg(label, i);
            if (label == n) {
                auto predict = forwardPropagation_predict(X, W, B);

                loss += get_loss(predict, label);
                correct += predict.maxer().second == label;
                ++count;
                
            }
        }

        // 0 
        // 2 0.89438   780
        // 3 0.91      618
        // 4 0.92      600
        cout << "==================" << endl;
        cout << n << endl;
        cout << count << endl;
        cout << correct / (double)count << endl;
        cout << loss / count << endl;
    }

}
