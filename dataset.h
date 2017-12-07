#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>


#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

struct MNISTDataSet
{
    void Init(const char* dataFile, const char* labelFile);

    vector<VectorXd> data;
    vector<uint8_t> labels;

    uint32_t numImgs;
    uint32_t imgSize;
};