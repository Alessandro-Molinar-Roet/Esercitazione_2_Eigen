#include <iostream>
#include <iomanip>
#include "Eigen/Eigen"

using namespace Eigen;
using namespace std;

// solve the linear system with PA = Lu factorization with partial pivoting
Vector2d Lu_system(Matrix2d const &M, Vector2d const &v)
{
    Vector2d x = M.partialPivLu().solve(v);
    return x;
}

// solve the linear system with QR = A factorization with no pivoting
Vector2d Qr_sytsem(Matrix2d const &M, Vector2d const &v)
{
    Vector2d x = M.householderQr().solve(v);
    return x;
}

int main()
{
    const Vector2d sol(-1.0e+0, -1.0e+00);

    // sistem 1
    Matrix2d A { // construct a 2x2 matrix
        {5.547001962252291e-01, -3.770900990025203e-02}, // first row
        {8.320502943378437e-01, -9.992887623566787e-01} // second row
    };
    Vector2d a(-5.169911863249772e-01, 1.672384680188350e-01); // column vector

    // sistem 2
    Matrix2d  B {
        {5.547001962252291e-01, -5.540607316466765e-01},
        {8.320502943378437e-01, -8.324762492991313e-01}
    };
    Vector2d b(-6.394645785530173e-04, 4.259549612877223e-04);

    // sistem 3
    Matrix2d C {
        {5.547001962252291e-01, -5.547001955851905e-01},
        {8.320502943378437e-01, -8.320502947645361e-01}
    };
    Vector2d c(-6.400391328043042e-10, 4.266924591433963e-10);

    // Palu
    Vector2d system1_palu= Lu_system(A,a);
    double system1_palu_error = (sol-system1_palu).norm()/sol.norm();
    Vector2d system2_palu= Lu_system(B,b);
    double system2_palu_error = (sol-system2_palu).norm()/sol.norm();
    Vector2d system3_palu= Lu_system(C,c);
    double system3_palu_error = (sol-system3_palu).norm()/sol.norm();

    // Qr
    Vector2d system1_qr = Qr_sytsem(A,a);
    double system1_qr_error = (sol-system1_qr).norm()/sol.norm();
    Vector2d system2_qr = Qr_sytsem(B,b);
    double system2_qr_error = (sol-system2_qr).norm()/sol.norm();
    Vector2d system3_qr = Qr_sytsem(C,c);
    double system3_qr_error = (sol-system3_qr).norm()/sol.norm();

    // print
    cout << fixed;
    cout << setprecision(16);
    cout << "First system solution    palu: [" << system1_palu.transpose()<< "]'  qr: [" << system1_qr.transpose() << "]'\n";
    cout << "Second system solution   palu: [" << system2_palu.transpose()<< "]'  qr: [" << system2_qr.transpose() << "]'\n";
    cout << "Third system solution    palu: [" << system3_palu.transpose()<< "]'  qr: [" << system3_qr.transpose() << "]'\n";
    cout << scientific << "\n";
    cout << "First system:    palu error " << system1_palu_error << "  qr error " << system1_qr_error << "\n";
    cout << "Second system:   palu error " << system2_palu_error << "  qr error " << system2_qr_error << "\n";
    cout << "Third system:    palu error " << system3_palu_error << "  qr error " << system3_qr_error << "\n";

    return 0;
}
