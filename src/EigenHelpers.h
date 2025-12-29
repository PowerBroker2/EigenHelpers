#pragma once
#include "Arduino.h"
#include "eigen.h"
#include <Eigen/Dense>

using namespace Eigen;

/**
 * @brief Make a skew-symmetric matrix from a 3-element vector.
 * Skew-symmetric matrices are often used to represent cross products.
 * 
 * @tparam Scalar float or double
 * @param w 3-element vector
 * @return 3x3 skew-symmetric matrix
 */
template<typename Scalar>
Matrix<Scalar, 3, 3> skew(const Matrix<Scalar, 3, 1>& w)
{
    Matrix<Scalar, 3, 3> C;
    C <<  0,    -w(2),  w(1),
         w(2),   0,    -w(0),
        -w(1),  w(0),   0;
    return C;
}

/**
 * @brief Compute covariance matrix of a data matrix (NxM, N = dof, M = samples)
 * 
 * @tparam Derived Any Eigen matrix type
 * @param mat Input matrix (NxM)
 * @return Covariance matrix (NxN)
 */
template<typename Derived>
Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::RowsAtCompileTime> cov(const Eigen::MatrixBase<Derived>& mat)
{
    using Scalar = typename Derived::Scalar;
    Vector<Scalar, Derived::RowsAtCompileTime> meanVec = mat.rowwise().mean();
    meanVec.transposeInPlace();
    auto centered = mat.colwise() - meanVec;
    return (centered * centered.adjoint()) / Scalar(mat.cols() - 1);
}

/**
 * @brief Compute square root of a 3x3 matrix (mat = V*D*V^-1)
 * 
 * @tparam Scalar float or double
 * @param mat 3x3 matrix
 * @return 3x3 square root of the matrix
 */
template<typename Scalar>
Matrix<Scalar, 3, 3> sqrtm(const Matrix<Scalar, 3, 3>& mat)
{
    EigenSolver<Matrix<Scalar, 3, 3>> es(mat);
    Matrix<Scalar, 3, 1> vals = es.eigenvalues().real();
    Matrix<Scalar, 3, 3> vecs = es.eigenvectors().real();
    
    Matrix<Scalar, 3, 3> sqrtVals = Matrix<Scalar, 3, 3>::Zero();
    sqrtVals.diagonal() << std::sqrt(vals(0)), std::sqrt(vals(1)), std::sqrt(vals(2));
    
    return vecs * sqrtVals * vecs.inverse();
}

/**
 * @brief Print a vector of any scalar type to a Stream
 */
template<typename VecType>
void printVec(const VecType& vec, int p, Stream& stream)
{
    for (int i = 0; i < vec.rows(); ++i)
    {
        if (vec(i) >= 0)
            stream.print(' ');
        stream.println(vec(i), p);
    }
}

/**
 * @brief Print a quaternion of any scalar type to a Stream
 */
template<typename Scalar>
void printQuat(const Quaternion<Scalar>& quat, int p, Stream& stream)
{
    stream.print("x: "); stream.println(quat.x(), p);
    stream.print("y: "); stream.println(quat.y(), p);
    stream.print("z: "); stream.println(quat.z(), p);
    stream.print("w: "); stream.println(quat.w(), p);
}

/**
 * @brief Print a matrix of any scalar type to a Stream
 */
template<typename MatType>
void printMat(const MatType& mat, int p, Stream& stream)
{
    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols(); ++j)
        {
            if (mat(i, j) >= 0)
                stream.print(' ');
            stream.print(mat(i, j), p);
            if (j != mat.cols() - 1)
                stream.print(", ");
        }
        stream.println();
    }
}
