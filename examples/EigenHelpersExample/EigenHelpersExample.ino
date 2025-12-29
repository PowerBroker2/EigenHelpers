#include "EigenHelpers.h"
#include "eigen.h"
#include <Eigen/Dense>




using namespace Eigen;




void setup()
{
    Serial.begin(115200);
}




void loop()
{
    Matrix3d mat;
    mat << 1.1, 2.2, 3.3,
           4.4, 5.5, 6.6,
           7.7, 8.8, 9.9;
    auto covMat = cov(mat);

    Serial.println("mat");
    printMat(mat, 1, Serial);
    Serial.println();
    Serial.println("covMat");
    printMat(covMat, 1, Serial);
    Serial.println();
    Serial.println();

    delay(1000);
}