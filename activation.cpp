#pragma once

#include<iostream>
#include<vector>
#include<math.h>
#include "matrix.cpp"
using namespace std;

float d_tanh(float x) {
    double t = tanh(x);
    return 1.f - t*t;
}

MatrixLib::Matrix softmax(const MatrixLib::Matrix& m) {
    if (m.empty()) return MatrixLib::Matrix();
    MatrixLib::Matrix res(m);

    double expSum = 0;
    
    float maxNum = res[0][0];
    for (int r = 0; r < m.rowSize(); ++r) {
        for (int c = 0; c < m.colSize(); ++c) {
            maxNum = max(maxNum, res[r][c]);
        }
    }
    for (int r = 0; r < m.rowSize(); ++r) {
        for (int c = 0; c < m.colSize(); ++c) {
            double expx = exp(res[r][c] - maxNum);
            expSum += expx;
            res[r][c] = expx;
        }
    }
    for (int r = 0; r < m.rowSize(); ++r) {
        for (int c = 0; c < m.colSize(); ++c) {
            res[r][c] /= expSum;
        }
    }
    return res;
}
MatrixLib::Matrix d_softmax(const MatrixLib::Matrix& m) {
    if (m.empty()) return MatrixLib::Matrix();
    MatrixLib::Matrix res(m);

    double expSum = 0;
    
    float maxNum = res[0][0];
    for (int r = 0; r < m.rowSize(); ++r) {
        for (int c = 0; c < m.colSize(); ++c) {
            maxNum = max(maxNum, res[r][c]);
        }
    }

    for (int r = 0; r < m.rowSize(); ++r) {
        for (int c = 0; c < m.colSize(); ++c) {
            double expx = exp(res[r][c] - maxNum);
            expSum += expx;
        }
    }

    for (int r = 0; r < m.rowSize(); ++r) {
        for (int c = 0; c < m.colSize(); ++c) {
            double expx = exp(res[r][c] - maxNum);
            res[r][c] = expx * (expSum - expx) / (expSum * expSum);
        }
    }
    return res;
}

MatrixLib::Matrix tanh(const MatrixLib::Matrix& m) {
    if (m.empty()) return MatrixLib::Matrix();
    MatrixLib::Matrix res(m);

    for (int r = 0; r < m.rowSize(); ++r) {
        for (int c = 0; c < m.colSize(); ++c) {
            res[r][c] = tanh(res[r][c]);
        }
    }

    return res;
}
MatrixLib::Matrix d_tanh(const MatrixLib::Matrix& m) {
    if (m.empty()) return MatrixLib::Matrix();
    MatrixLib::Matrix res(m);

    for (int r = 0; r < m.rowSize(); ++r) {
        for (int c = 0; c < m.colSize(); ++c) {
            res[r][c] = d_tanh(res[r][c]);
        }
    }

    return res;
}

MatrixLib::Matrix sigmoid(const MatrixLib::Matrix& m) {
    if (m.empty()) return MatrixLib::Matrix();
    MatrixLib::Matrix res(m);

    for (int r = 0; r < m.rowSize(); ++r) {
        for (int c = 0; c < m.colSize(); ++c) {
            res[r][c] = 1.f / (1.f + exp(-res[r][c]));
        }
    }

    return res;
}
MatrixLib::Matrix d_sigmoid(const MatrixLib::Matrix& m) {
    if (m.empty()) return MatrixLib::Matrix();
    MatrixLib::Matrix res(m);

    for (int r = 0; r < m.rowSize(); ++r) {
        for (int c = 0; c < m.colSize(); ++c) {
            double sigmoidNum = 1.f / (1 + exp(-res[r][c]));
            res[r][c] = sigmoidNum * (1.f - sigmoidNum);
        }
    }

    return res;
}


MatrixLib::Matrix linear(const MatrixLib::Matrix& m) {
    return m;
}
MatrixLib::Matrix d_linear(const MatrixLib::Matrix& m) {
    return MatrixLib::Matrix(m.rowSize(), m.colSize(), 1);
}



MatrixLib::Matrix relu(const MatrixLib::Matrix& m) {
    if (m.empty()) return MatrixLib::Matrix();
    MatrixLib::Matrix res(m);

    for (int r = 0; r < m.rowSize(); ++r) {
        for (int c = 0; c < m.colSize(); ++c) {
            res[r][c] = max(res[r][c], 0.f);
        }
    }

    return res;
}

MatrixLib::Matrix d_relu(const MatrixLib::Matrix& m) {
    if (m.empty()) return MatrixLib::Matrix();
    MatrixLib::Matrix res(m);

    for (int r = 0; r < m.rowSize(); ++r) {
        for (int c = 0; c < m.colSize(); ++c) {
            res[r][c] = (res[r][c] > 0 ? 1 : 0);
        }
    }

    return res;
}


MatrixLib::Matrix leaky_relu(const MatrixLib::Matrix& m) {
    if (m.empty()) return MatrixLib::Matrix();
    MatrixLib::Matrix res(m);

    for (int r = 0; r < m.rowSize(); ++r) {
        for (int c = 0; c < m.colSize(); ++c) {
            res[r][c] = (res[r][c] > 0 ? res[r][c] : 0.33 * res[r][c]);
        }
    }

    return res;
}

MatrixLib::Matrix d_leaky_relu(const MatrixLib::Matrix& m) {
    if (m.empty()) return MatrixLib::Matrix();
    MatrixLib::Matrix res(m);

    for (int r = 0; r < m.rowSize(); ++r) {
        for (int c = 0; c < m.colSize(); ++c) {
            res[r][c] = (res[r][c] > 0 ? 1 : 0.33f);
        }
    }

    return res;
}
