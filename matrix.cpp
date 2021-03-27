#pragma once

#include<iostream>
#include<vector>
#include<time.h>
#include<math.h>
using namespace std;

namespace MatrixLib {
    float randFloat(float, float);
    void randInit();

    class Matrix {
        
        vector<vector<float>> matrix;

    public:

        Matrix() {}
        Matrix(int rows, int cols, float initVal = 0);
        Matrix(int rows, int cols, float lower, float upper);
        Matrix(const Matrix& another);
        Matrix(const vector<vector<float>>& vecvec);
        Matrix(const vector<float>& vec, int rowFold = 1);
        Matrix(const vector<unsigned char>& vec, int rowFold = 1);
        Matrix(const string& serialzed);

        vector<float>& operator[](int row);
        void operator+=(const Matrix& another);
        void operator+=(float num);
        void operator-=(const Matrix& another);
        void operator-=(float num);
        void operator*=(float num);
        void operator*=(const Matrix& another);
        void operator/=(float num);

        void operator=(const Matrix& another);
        Matrix operator+(const Matrix& another);
        Matrix operator+(float num);
        Matrix operator-(const Matrix& another);
        Matrix operator-(float num);
        Matrix operator*(float num);
        Matrix operator*(const Matrix& another);
        Matrix operator/(float num);

        Matrix sub(int lrow, int lcol, int rrow, int rcol);
        vector<float> getRows(int row);
        vector<float> getCols(int col);
        void unfold();


        void addRow(vector<float>& rowVec, int row = -1);
        void addCol(vector<float>& colVec, int col = -1);
        void fillRow(vector<float>& rowVec, int row);
        void fillCol(vector<float>& colVec, int col);
        void extendRow(int amount, float initVal = 0);
        void extendCol(int amount, float initVal = 0);
        void shrinkRow(int amount);
        void shrinkCol(int amount);
        void reshape(int newRow, int newCol, float defaultVal = 0);
        void fillAll(vector<float>& vec);

        float norm() const;
        float norm2() const;
        void transpose();
        float dotALL(const Matrix& another) const;
        Matrix dotElem(const Matrix& another) const;
        Matrix dotRow(const Matrix& another) const;


        bool empty() const;
        int rowSize() const;
        int colSize() const;
        float at(int row, int col) const;
        pair<int, int> maxer() const;
        pair<int, int> miner() const;
        float maxval() const;
        float minval() const;

        float sum() const;

        void max_pooling(int poolSize);
        float average();
        
        double std(double avg);

        string serialize();

        void print1() const;
        void print2(int eachRows = 10) const;
        void print3() const;
    };

    void randInit(){
        srand(unsigned(time(0)));
    }

    float randFloat(float lower, float upper) {
        // -1.5, 2.5
        // w=4 [0, 4]

        // 0, 32767 = [0, 1]
        return (rand() / 32767.f) * (upper - lower) + lower;
    }

    Matrix unitMatrix(int n) {
        Matrix res(n, n);
        for (int i = 0; i < n; ++i) {
            res[i][i] = 1;
        }
        return res;
    }

    float norm(const Matrix& m) {
        return m.norm();
    }

    float dotALL(const Matrix& m1, const Matrix& m2) {
        return m1.dotALL(m2);
    }

    Matrix dotElem(const Matrix& m1, const Matrix& m2) {
        return m1.dotElem(m2);
    }

    Matrix dotRow(const Matrix& m1, const Matrix& m2) {
        return m1.dotRow(m2);
    }


    Matrix transpose(const Matrix& m) {
        Matrix res(m.colSize(), m.rowSize());
        for (int r = 0; r < m.rowSize(); ++r) {
            for (int c = 0; c < m.colSize(); ++c) {
                res[c][r] = m.at(r, c);
            }
        }
        return res;
    }

    vector<float> unfold(const Matrix& m) {
        if (m.empty()) return {};
        vector<float> res; res.reserve(m.rowSize() * m.colSize());

        for (int r = 0; r < m.rowSize(); ++r) {
            for (int c = 0; c < m.colSize(); ++c) {
                res.emplace_back(m.at(r, c));
            }
        }

        return res;
    }

    pair<int, int> maxer(const Matrix& m) {
        return m.maxer();
    }
    pair<int, int> miner(const Matrix& m) {
        return m.miner();
    }
    float maxval(const Matrix& m) {
        return m.maxval();
    }
    float minval(const Matrix& m) {
        return m.minval();
    }
    
    int maxer(const vector<float>& vec) {
        int idx = -1;
        float res = INT32_MIN;
        for (int i = 0; i < vec.size(); ++i) {
            if (res < vec[i]) {
                idx = i;
                res = vec[i];
            }
        }

        return idx;
    }

    int miner(const vector<float>& vec) {
        int idx = -1;
        float res = INT32_MAX;
        for (int i = 0; i < vec.size(); ++i) {
            if (res > vec[i]) {
                idx = i;
                res = vec[i];
            }
        }

        return idx;
    }

    float maxval(const vector<float>& vec) {
        float res = INT32_MIN;
        for (int i = 0; i < vec.size(); ++i) {
            res = max(res, vec[i]);
        }

        return res;
    }

    float minval(const vector<float>& vec) {
        float res = INT32_MAX;
        for (int i = 0; i < vec.size(); ++i) {
            res = min(res, vec[i]);
        }

        return res;
    }

    float sum(const Matrix& m) {
        return m.sum();
    }



    Matrix::Matrix(int rows, int cols, float initVal) {
        matrix.resize(rows, vector<float>(cols, initVal));
    }
    Matrix::Matrix(int rows, int cols, float lower, float upper) {
        matrix.resize(rows, vector<float>(cols));
        
        for (int r = 0; r < rows; ++r) {
            
            for (int c = 0; c < cols; ++c) {
                matrix[r][c] = randFloat(lower, upper);
            }
        }
    }
    Matrix::Matrix(const Matrix& another) {
        this->matrix = another.matrix;
    }
    Matrix::Matrix(const vector<vector<float>>& vecvec) {
        matrix = vecvec;
    }
    Matrix::Matrix(const vector<float>& vec, int rowFold) {
        int len = vec.size();
        int eachFold = (len + rowFold - 1) / rowFold;
        int ptr = 0;
        for (int i = 0; i < rowFold; ++i) {
            int count = eachFold;
            matrix.push_back({});
            while (count-- > 0) {
                matrix.back().emplace_back(ptr < len ? vec[ptr] : 0);
                ++ptr;
            }
        }
    }
    Matrix::Matrix(const vector<unsigned char>& vec, int rowFold) {
        int len = vec.size();
        int eachFol = (len + rowFold - 1) / rowFold;
        int ptr = 0;
        for (int i = 0; i < rowFold; ++i) {
            int count = eachFol;
            matrix.push_back({});
            while (count-- > 0) {
                matrix.back().emplace_back(ptr < len ? (float)vec[ptr] : 0);
                ++ptr;
            }
        }
    }
    Matrix::Matrix(const string& serialzed) {
        if (serialzed.size() < 2 || serialzed[0] != '!' || serialzed[1] != 'M') return;
        // !M,R,C,[e1,e2...],[...],[...],[...],...[...]@
        //            ________________________________
        //                          R
        //            __________    
        //                C
        int ptr = 3;

        int R = 0;
        int C = 0;
        while (serialzed[ptr] != ',') {
            R *= 10;
            R += serialzed[ptr] - '0';
            ++ptr;
        }
        ++ptr;
        while (serialzed[ptr] != ',') {
            C *= 10;
            C += serialzed[ptr] - '0';
            ++ptr;
        }
        ++ptr; // [
        ++ptr; // e1

        this->matrix.resize(R, vector<float>(C));

        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                int begin = ptr;
                while (serialzed[ptr] != ',' && serialzed[ptr] != ']') {
                    ++ptr;
                }
                this->matrix[r][c] = stof(serialzed.substr(begin, ptr-begin));

                ++ptr;
            }
            if (serialzed[ptr] == '@') break;
            ptr += 2;
        }
    }


    vector<float>& Matrix::operator[](int row) {
        return matrix[row];
    }

    void Matrix::operator+=(const Matrix& another) {
        for (int r = 0; r < min(matrix.size(), another.matrix.size()); ++r) {
            for (int c = 0; c < min(matrix[0].size(), another.matrix[0].size()); ++c) {
                this->matrix[r][c] += another.matrix[r][c];
            }
        }
    }
    void Matrix::operator+=(float num) {
        for (int r = 0; r < matrix.size(); ++r) {
            for (int c = 0; c < matrix[0].size(); ++c) {
                this->matrix[r][c] += num;
            }
        }
    }
    void Matrix::operator-=(const Matrix& another) {
        for (int r = 0; r < min(matrix.size(), another.matrix.size()); ++r) {
            for (int c = 0; c < min(matrix[0].size(), another.matrix[0].size()); ++c) {
                this->matrix[r][c] -= another.matrix[r][c];
            }
        }
    }
    void Matrix::operator-=(float num) {
        for (int r = 0; r < matrix.size(); ++r) {
            for (int c = 0; c < matrix[0].size(); ++c) {
                this->matrix[r][c] -= num;
            }
        }
    }
    void Matrix::operator*=(float num) {
        for (int r = 0; r < matrix.size(); ++r) {
            for (int c = 0; c < matrix[0].size(); ++c) {
                this->matrix[r][c] *= num;
            }
        }
    }
    void Matrix::operator*=(const Matrix& another) {
        matrix = (*this * another).matrix;
    }
    void Matrix::operator/=(float num) {
        if (num == 0) return;
        for (int r = 0; r < matrix.size(); ++r) {
            for (int c = 0; c < matrix[0].size(); ++c) {
                this->matrix[r][c] /= num;
            }
        }
    }

    void Matrix::operator=(const Matrix& another) {
        this->matrix = another.matrix;
    }
    Matrix Matrix::operator+(const Matrix& another) {
        Matrix res(*this);
        for (int r = 0; r < min(matrix.size(), another.matrix.size()); ++r) {
            for (int c = 0; c < min(matrix[0].size(), another.matrix[0].size()); ++c) {
                res.matrix[r][c] += another.matrix[r][c];
            }
        }
        return res;
    }
    Matrix Matrix::operator+(float num) {
        Matrix res(*this);
        for (int r = 0; r < matrix.size(); ++r) {
            for (int c = 0; c < matrix[0].size(); ++c) {
                res.matrix[r][c] += num;
            }
        }
        return res;
    }
    Matrix Matrix::operator-(const Matrix& another) {
        Matrix res(*this);
        for (int r = 0; r < min(matrix.size(), another.matrix.size()); ++r) {
            for (int c = 0; c < min(matrix[0].size(), another.matrix[0].size()); ++c) {
                res.matrix[r][c] -= another.matrix[r][c];
            }
        }
        return res;
    }
    Matrix Matrix::operator-(float num) {
        return *this + (-num);
    }
    Matrix Matrix::operator*(float num) {
        Matrix res(*this);
        for (int r = 0; r < matrix.size(); ++r) {
            for (int c = 0; c < matrix[0].size(); ++c) {
                res.matrix[r][c] *= num;
            }
        }
        return res;
    }
    Matrix Matrix::operator*(const Matrix& another) {
        int R1 = matrix.size();
        if (R1 == 0) return *this;
        int C1 = matrix[0].size();
        int R2 = another.matrix.size();
        if (R2 == 0) return *this;
        int C2 = another.matrix[0].size();

        if (C1 != R2) return *this;

        Matrix res(R1, C2);

        for (int r = 0; r < R1; ++r) {
            for (int c = 0; c < C2; ++c) {
                double sum = 0;
                for (int i = 0; i < C1; ++i) {
                    sum += this->matrix[r][i] * another.matrix[i][c];
                }

                res[r][c] = sum;
            }
        }

        return res;
    }   
    Matrix Matrix::operator/(float num) {
        if (num == 0) return *this;
        return *this * (1.f / num);
    }

    Matrix Matrix::sub(int lrow, int lcol, int rrow, int rcol) {
        if (lrow >= rrow || lcol >= rcol) return Matrix();
        vector<vector<float>> res(rrow-lrow, vector<float>(rcol-lcol));
        for (int r = lrow; r < rrow; ++r) {
            for (int c = lcol; c < rcol; ++c) {
                res[r-lrow][c-lcol] = matrix[r][c];
            }
        }
        return res;
    }
    vector<float> Matrix::getRows(int row) {
        return matrix[row];
    }
    vector<float> Matrix::getCols(int col) {
        vector<float> colVec; colVec.reserve(matrix.size());
        for (int r = 0; r < matrix.size(); ++r) {
            colVec.emplace_back(matrix[r][col]);
        }
        return colVec;
    }
    void Matrix::unfold() {
        if (matrix.empty()) return;
        int R = this->rowSize();
        int C = this->colSize();

        for (int r = 1; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                matrix[0].emplace_back(matrix[r][c]);
            }
        }
        matrix.resize(1);
    }


    void Matrix::addRow(vector<float>& rowVec, int row) {
        if (row == -1) row = matrix.size();
        auto it = matrix.insert(matrix.begin() + row, rowVec);
        if (matrix.size()) it->resize(matrix[0].size(), 0);

    }
    void Matrix::addCol(vector<float>& colVec, int col) {
        if (matrix.empty()) {
            for (int num : colVec) {
                matrix.push_back({});
                matrix.back().emplace_back(num);
            }
            return;
        }
        if (col == -1) col = matrix[0].size();

        for (int r = 0; r < matrix.size(); ++r) {
            matrix[r].insert(matrix[r].begin()+col, r < colVec.size() ? colVec[r] : 0);
        }

    }
    void Matrix::fillRow(vector<float>& rowVec, int row) {
        for (int c = 0; c < matrix[0].size(); ++c) {
            matrix[row][c] = (c < rowVec.size() ? rowVec[c] : 0);
        }
    }
    void Matrix::fillCol(vector<float>& colVec, int col) {
        for (int r = 0; r < matrix.size(); ++r) {
            matrix[r][col] = (r < colVec.size() ? colVec[r] : 0);
        }
    }
    void Matrix::extendRow(int amount, float initVal) {
        if (matrix.empty()) return;
        while (amount-- > 0) matrix.emplace_back(vector<float>(matrix[0].size(), initVal));
    }
    void Matrix::extendCol(int amount, float initVal) {
        if (matrix.empty()) return;
        for (int r = 0; r < matrix.size(); ++r) {
            matrix[r].resize(matrix[r].size() + amount, initVal);
        }
    }
    void Matrix::shrinkRow(int amount) {
        matrix.resize(rowSize() - amount);
        // while (amount-- > 0) matrix.pop_back();
    }
    void Matrix::shrinkCol(int amount) {
        for (int r = 0; r < matrix.size(); ++r) {
            matrix[r].resize(colSize() - amount);
            // for (int t = 0; t < amount; ++t) matrix[r].pop_back();
        }
    }
    void Matrix::reshape(int newRow, int newCol, float defaultVal) {
        Matrix newm(newRow, newCol, false, defaultVal);
        this->unfold();
        newm.fillAll(this->matrix.front());
        this->matrix = newm.matrix;
    }
    void Matrix::fillAll(vector<float>& vec) {
        int ptr = 0;
        for (int r = 0; r < matrix.size(); ++r) {
            for (int c = 0; c < matrix[0].size(); ++c) {
                if (ptr < vec.size()) matrix[r][c] = vec[ptr++];
                else return;
            }
        }
    }

    float Matrix::norm() const {
        double res = 0;
        for (int r = 0; r < matrix.size(); ++r) {
            for (int c = 0; c < matrix[0].size(); ++c) {
                res += matrix[r][c] * matrix[r][c];
            }
        }
        return sqrt(res);
    }
    float Matrix::norm2() const {
        double res = 0;
        for (int r = 0; r < matrix.size(); ++r) {
            for (int c = 0; c < matrix[0].size(); ++c) {
                res += matrix[r][c] * matrix[r][c];
            }
        }
        return res;
    }
    void Matrix::transpose() {
        vector<vector<float>> res(this->colSize(), vector<float>(this->rowSize()));
        for (int r = 0; r < matrix.size(); ++r) {
            for (int c = 0; c < matrix[0].size(); ++c) {
                res[c][r] = matrix[r][c];
            }
        }
        matrix = res;
    }
    float Matrix::dotALL(const Matrix& another) const {
        double res = 0;
        for (int r = 0; r < min(matrix.size(), another.matrix.size()); ++r) {
            for (int c = 0; c < min(matrix[0].size(), another.matrix[0].size()); ++c) {
                res += another.matrix[r][c] * this->matrix[r][c];
            }
        }
        return res;
    }
    Matrix Matrix::dotElem(const Matrix& another) const{
        Matrix res(*this);
        for (int r = 0; r < min(matrix.size(), another.matrix.size()); ++r) {
            for (int c = 0; c < min(matrix[0].size(), another.matrix[0].size()); ++c) {
                res.matrix[r][c] *= another.matrix[r][c];
            }
        }
        return res;
    }
    Matrix Matrix::dotRow(const Matrix& another) const {
        if (this->colSize() != another.colSize()) return Matrix();
        Matrix res(this->rowSize(), 1);
        for (int r = 0; r < this->rowSize(); ++r) {
            double sum = 0;
            for (int c = 0; c < this->colSize(); ++c) {
                sum += this->matrix[r][c] * another.matrix[0][c];
            }
            res[r][0] = sum;
        }
        return res;
    }


    bool Matrix::empty() const {
        return this->matrix.size() == 0;
    }
    int Matrix::rowSize() const {
        return matrix.size();
    }
    int Matrix::colSize() const {
        return rowSize() ? matrix[0].size() : 0;
    }
    float Matrix::at(int row, int col) const {
        return matrix[row][col];
    }
    pair<int, int> Matrix::maxer() const {
        float res = INT32_MIN;
        int rr, rc;
        for (int r = 0; r < matrix.size(); ++r) {
            for (int c = 0; c < matrix[0].size(); ++c) {
                if (matrix[r][c] > res) {
                    rr = r;
                    rc = c;
                    res = matrix[r][c];
                }
            }
        }
        return {rr, rc};
    }
    pair<int, int> Matrix::miner() const {
        float res = INT32_MIN;
        int rr, rc;
        for (int r = 0; r < matrix.size(); ++r) {
            for (int c = 0; c < matrix[0].size(); ++c) {
                if (matrix[r][c] < res) {
                    rr = r;
                    rc = c;
                    res = matrix[r][c];
                }
            }
        }
        return {rr, rc};
    }
    float Matrix::maxval() const {
        float res = INT32_MIN;
        for (int r = 0; r < matrix.size(); ++r) {
            for (int c = 0; c < matrix[0].size(); ++c) {
                if (matrix[r][c] > res) {
                    res = matrix[r][c];
                }
            }
        }
        return res;
    }
    float Matrix::minval() const {
        float res = INT32_MAX;
        for (int r = 0; r < matrix.size(); ++r) {
            for (int c = 0; c < matrix[0].size(); ++c) {
                if (matrix[r][c] < res) {
                    res = matrix[r][c];
                }
            }
        }
        return res;
    }

    float Matrix::sum() const {
        double res = 0;
        for (int r = 0; r < matrix.size(); ++r) {
            for (int c = 0; c < matrix[0].size(); ++c) {
                res += matrix[r][c];
            }
        }
        return res;
    }

    void Matrix::print1() const {
        cout << "Matrix===========================================" << endl;
        cout << "Rows = " << matrix.size() << " , Cols = " << (matrix.size() ? matrix[0].size() : 0) << endl;
        if (!matrix.size()) return;
        cout << '\\' << '\t';
        for (int c = 0; c < matrix[0].size(); ++c) {
            cout << c << '\t';
        }
        cout << endl;

        for (int r = 0; r < matrix.size(); ++r) {

            int count = 0;
            for (int c = 0; c < matrix[0].size(); ++c) {
                if (count == 0) {
                    cout << r << '\t';
                    ++count;
                }
                cout << matrix[r][c] << '\t';
            }
            cout << endl;
        }

        cout << endl;
    }
    void Matrix::print2(int eachRows) const {
        cout << "Matrix===========================================" << endl;
        cout << "Rows = " << matrix.size() << " , Cols = " << (matrix.size() ? matrix[0].size() : 0) << endl;
        
        int count = 0;
        for (int r = 0; r < matrix.size(); ++r) {
            cout << '[';
            for (int c = 0; c < matrix[0].size(); ++c) {
                cout << matrix[r][c] << ", ";
                ++count;
                if (count == eachRows) {
                    cout << endl;
                    count = 0;
                }
            }
            cout << ']' << endl;
            count = 0;
        }

        
    }
    void Matrix::print3() const {
        cout << "Matrix Show===========================================" << endl;
        cout << "Rows = " << matrix.size() << " , Cols = " << (matrix.size() ? matrix[0].size() : 0) << endl;

        for (int r = 0; r < matrix.size(); ++r) {
            for (int c = 0; c < matrix[0].size(); ++c) {

                if (matrix[r][c] > 1e-7) {
                    cout << "* ";
                }
                else {
                    cout << "  ";
                }

            }
            cout << endl;
        }
    }

    string Matrix::serialize() {
        int R = this->rowSize();
        int C = this->colSize();

        // !M,R,C,[e1,e2...],[...],[...],[...]...[...]@
        //            ________________________________
        //                          R
        //            __________    
        //                C


        string data = "!M," + to_string(R) + "," + to_string(C) + ",";

        for (int r = 0; r < R; ++r) {
            data += '[';
            for (int c = 0; c < C; ++c) {
                data += to_string(matrix[r][c]);  
                data += ',';
            }
            data.pop_back();
            data += "],";
        }

        data.pop_back();
        data += '@';

        return data;
    }
    void Matrix::max_pooling(int poolSize) {
        int R = this->rowSize();
        int C = this->colSize();
        if (R < poolSize || C < poolSize) return;
        int newR = ceil((R - poolSize) / (float)poolSize) + 1;
        int newC = ceil((C - poolSize) / (float)poolSize) + 1;

        for (int r = 0; r < newR; ++r) {
            for (int c = 0; c < newC; ++c) {
                
                float maxNum = -256.f;
                for (int i = r*poolSize; i < min(R, (r+1)*poolSize); ++i) {
                    for (int j = c*poolSize; j < min((c+1)*poolSize, C); ++j) {
                        maxNum = max(matrix[i][j], maxNum);
                    }
                }
                
                matrix[r][c] = maxNum;
            }
        }

        this->shrinkRow(R-newR);
        this->shrinkCol(C-newC);
    }
    float Matrix::average() {
        int R = this->rowSize();
        int C = this->colSize();
        double res = 0;
        
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                res += matrix[r][c];
            }
        }

        return res / (R * C);
    }
    double Matrix::std(double avg) {
        int R = this->rowSize();
        int C = this->colSize();
        double res = 0;
        
        for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                res += pow(matrix[r][c] - avg, 2);
            }
        }

        return sqrt(res / (R * C));
    }

};