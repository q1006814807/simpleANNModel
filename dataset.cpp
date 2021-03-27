#pragma once
#include<iostream>
#include<fstream>
#include<vector>
#include "matrix.cpp"
using namespace std;
using namespace MatrixLib;

class Dataset {
    const string path_training_image = "mnist\\train-images.idx3-ubyte";
    const string path_training_label = "mnist\\train-labels.idx1-ubyte";
    const string path_test_image     = "mnist\\t10k-images.idx3-ubyte";
    const string path_test_label     = "mnist\\t10k-labels.idx1-ubyte";
    
    const int image_size = 28*28;
    const int label_size = 1;
    const int training_image_start = 16;
    const int training_label_start = 8;
    const int test_image_start = 16;
    const int test_label_start = 8;


    int training_amount, test_amount;

    vector<Matrix> trainingImageSet;
    vector<int> trainingLabelSet;
    vector<Matrix> testImageSet;
    vector<int> testLabelSet;

    bool isTrainingInit = false;
    bool isTestInit = false;

public:
    void initTrainingDataset() {
        cout << "dataset start loading..." << endl;
        if (!isTrainingInit)
            loadDataSet(path_training_image, 
                        path_training_label, 
                        training_image_start, 
                        training_label_start,
                        training_amount,
                        trainingImageSet,
                        trainingLabelSet
                        );
        isTrainingInit = true;
        cout << "dataset load success!" << endl;
    }
    void initTestDataset() {
        cout << "dataset start loading..." << endl;
        if (!isTestInit)
            loadDataSet(path_test_image, 
                        path_test_label, 
                        test_image_start, 
                        test_label_start,
                        test_amount,
                        testImageSet,
                        testLabelSet
                        );
        isTestInit = true;
        cout << "dataset load success!" << endl;
    }

    void init() {
        cout << "dataset start loading..." << endl;
        if (!isTrainingInit)
            loadDataSet(path_training_image, 
                        path_training_label, 
                        training_image_start, 
                        training_label_start,
                        training_amount,
                        trainingImageSet,
                        trainingLabelSet
                        );
        isTrainingInit = true;
        if (!isTestInit)
            loadDataSet(path_test_image, 
                        path_test_label, 
                        test_image_start, 
                        test_label_start,
                        test_amount,
                        testImageSet,
                        testLabelSet
                        );
        isTestInit = true;
        cout << "dataset load success!" << endl;
    }

    Dataset(int training_amount, int test_amount) {
        this->training_amount = training_amount;
        this->test_amount = test_amount;

        trainingImageSet.reserve(this->training_amount);
        trainingLabelSet.reserve(this->training_amount);
        testImageSet.reserve(this->training_amount);
        testLabelSet.reserve(this->training_amount);


    }

    void loadDataSet(    const string& path_image, 
                         const string& path_label,
                         const int image_start,
                         const int label_start,
                         const int amount,
                         vector<Matrix>& imageSet,
                         vector<int>& labelSet
                    ) {
        ifstream ifs;

        ifs.open(path_image, ios::in | ios::binary);
        if (!ifs.is_open()) return;

        unsigned char data1[image_size];
        ifs.seekg(image_start, ios::cur);

        for (int i = 0; i < amount; ++i) {
            // ifs.read((char*)&data1, sizeof(data1));
            // imageSet.emplace_back(Matrix(vector<unsigned char>({data1, data1+image_size}), 28));
            // imageSet.back().max_pooling(2);
            // imageSet.back().unfold();

            ifs.read((char*)&data1, sizeof(data1));
            imageSet.emplace_back(Matrix(vector<unsigned char>({data1, data1+image_size}), 28));
            imageSet.back().max_pooling(2);

            double avg = imageSet.back().average();
            imageSet.back() -= avg;
            imageSet.back() /= imageSet.back().std(avg);

            imageSet.back().unfold();
        }
        ifs.close();

        // read label
        ifs.open(path_label, ios::in | ios::binary);
        if (!ifs.is_open()) return;

        unsigned char data2[label_size];
        ifs.seekg(label_start, ios::cur);

        for (int i = 0; i < amount; ++i) {
            ifs.read((char*)&data2, sizeof(data2));
            labelSet.emplace_back((int)data2[0]);
        }
        ifs.close();
    }

    Matrix& readTrainingImg(int& label, int select) {
        label = trainingLabelSet[select];
        return trainingImageSet[select];
    }

    Matrix& readTestImg(int& label, int select) {
        label = testLabelSet[select];
        return testImageSet[select];
    }




};

// int main() {
//     int label = -1;
//     auto imgMatrix = FileOpen::readTrainingImg(label);
//     imgMatrix.reshape(28, 28);
    
//     imgMatrix.print3();
//     cout << label << endl;

//     system("pause");
// }