#include <iostream>
#include <math.h>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>

using namespace std;

// Performs multiplication between the tridiagonal matrix and the vector
vector<double> tridiagonal_multiplication(vector<vector<double>> M, vector<double> T);

// Build the tridiagonal matrix with Dirichlet boundary conditions
vector<vector<double>> build_tridiagonal(int n);

// Build the matrix of 3 vectors starting from the tridiagonal matrix
vector<vector<double>> build_trivector(vector<vector<double>> A);

// Gauss-Jordan method (forward elimination backward subsitution)
vector<double> gauss_jordan(vector<vector<double>> X, vector<double> b);

// Decomposes matrix into diagonal, upper triangular, lower triangular, in order to perform Jacobi iteration
void DLU_decomposition(vector<vector<double>> &A, vector<double> &D, vector<vector<double>> &L, vector<vector<double>> &U);

void Jacobi_iteration(vector<double> D, vector<vector<double>> L, vector<vector<double>> U, vector<double> &x, vector<double> b);

// Print 1D vector
void print_vector(vector<double> x);

// Print 2D vector (overrides 1D)
void print_vector(vector<vector<double>> X);

const double D = 0.5; // Difussion coefficient
const double eps = 1.0; // RHS member of the equation
const double T0 = 1.0; // Boundary conditions
const double L = 1.0; // Grid length
const int NGauss = 100;
const int N = 100; // Grid points
const double h = (double) 2*L/(N-1); // Integration step (-1 important!)
const double h_gauss = (double) 2*L/(NGauss-1);
const double n_iter_Jacobi = 50; // Jacobi iterations

ofstream gaussfile, jacobifile;

int main(){



    /******************** Setting up files for plotting afterwards ***************/
    gaussfile.open("gauss100.csv");
    jacobifile.open("jacobi100.csv");
    gaussfile << "x,sol" << endl;
    jacobifile << "val0";
    for (int i=1; i<N; i++) jacobifile << ",val" << i;
    jacobifile << endl;
    /*****************************************************************************/




    /******************** Test sparse matrix multiplication **********************/
    vector<vector<double>> Mtest(N);
    for(int i=0; i<N; i++){
        Mtest[i].resize(N, 0.0);
        Mtest[i][i-1] = 1;
        Mtest[i][i] = 2;
        Mtest[i][i+1] = 3;
    }
    vector<double> vtest(N, 1.0);
    vector<vector<double>> Mtest_red = build_trivector(Mtest);
    cout << "Test matrix" << endl;
    print_vector(Mtest);
    cout << "3-vector" << endl;
    print_vector(Mtest_red);
    cout << "Test vector" << endl;
    print_vector(vtest);
    cout << "Multiplication result" << endl;
    print_vector(tridiagonal_multiplication(Mtest_red, vtest));
    /*****************************************************************************/





/*                              Gauss-Jordan algorithm                            */




    /************** Set up matrices and vectors for the problem ******************/
    vector<vector<double>> A = build_tridiagonal(NGauss); // prepare matrix A such that Ax = b
    vector<vector<double>> M = build_trivector(A); // transform tridiagonal matrix A in a matrix M of three vectors containing the diagonals
    // Prepare RHS of the matrix equation
    vector<double> b(NGauss, 0.0);
    for(int i=1; i<NGauss-1; i++) b[i] = (double) -h_gauss*h_gauss/D*eps;
    b[0] = T0;
    b[NGauss-1] = T0;

    // Some printing
    cout << "Original matrix:" << endl;
    print_vector(A);
    cout << "Reduced matrix: "<< endl;
    print_vector(M);
    /*****************************************************************************/




    /********** Solve diffusion equation with with Gauss-Jordan algorithm ********/
    auto begin = chrono::high_resolution_clock::now();
    vector<double> sol1 = gauss_jordan(A, b);
    auto end = chrono::high_resolution_clock::now();
    auto deltaT_midpoint = chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
    cout << "Solution found. Time taken: " << scientific  << setprecision(3) << deltaT_midpoint*1e-9 << defaultfloat << "s." << endl << "The solution is:" << endl;
    print_vector(sol1);
    for(int i=0; i<NGauss; i++) gaussfile << -L + (double) i*h_gauss << "," << sol1[i] << endl;

    // Check if solution x is correct by calculating residuals Ax - b
    vector<double> estimated_b = tridiagonal_multiplication(M, sol1);
    cout << "Residuals: " << endl;
    for(int i=0; i<NGauss; i++) cout << estimated_b[i] - b[i] << " ";
    cout << endl << endl;
    /*****************************************************************************/
    




/*                              Jabcobi Iteration                                */





    /************** Set up matrices and vectors for the problem ******************/
    vector<vector<double>> A2 = build_tridiagonal(N); // prepare matrix A such that Ax = b
    vector<vector<double>> M2 = build_trivector(A2); // transform tridiagonal matrix A in a matrix M of three vectors containing the diagonals
    // Prepare RHS of the matrix equation
    vector<double> b2(N, 0.0);
    for(int i=1; i<N-1; i++) b2[i] = (double) -h*h/D*eps;
    b[0] = T0;
    b[N-1] = T0;

    // Some printing
    cout << "Original matrix:" << endl;
    print_vector(A);
    cout << "Reduced matrix: "<< endl;
    print_vector(M);
    /*****************************************************************************/



    /*********** Solve diffusion equation with Jacobi iteration method ***********/
    vector<double> diag(N, 0.0);
    vector<vector<double>> L_mat(N);
    vector<vector<double>> U_mat(N);
    vector<double> sol2(N, 0.0);
    for(int i=0; i<N; i++){
        L_mat[i].resize(N, 0.0);
        U_mat[i].resize(N, 0.0);
    }
    DLU_decomposition(A2, diag, L_mat, U_mat);
    cout << "Diagonal:" << endl;
    print_vector(diag);
    cout << "Lower triangular" << endl;
    print_vector(L_mat);
    cout << "Upper triangular" << endl;
    print_vector(U_mat);
    for(uint i=0; i<n_iter_Jacobi; i++) Jacobi_iteration(diag, L_mat, U_mat, sol2, b);
    
    estimated_b = tridiagonal_multiplication(M2, sol2);
    cout << "Residuals after " << n_iter_Jacobi << " steps:" << endl;
    for(int i=0; i<N; i++) cout << estimated_b[i] - b2[i] << " ";
    cout << endl << endl;
    /***************************************************************************/




    return 0;
}




// Performs multiplication between the tridiagonal matrix and the vector
vector<double> tridiagonal_multiplication(vector<vector<double>> M, vector<double> T){
    uint n = T.size();
    vector<double> res(n, 0.0);
    for(uint i=1; i<n-1; i++){
        res[i] = M[i][0]*T[i-1] + M[i][1]*T[i] + M[i][2]*T[i+1];
    }
    res[0] = M[0][0]*T[n-1] + M[0][1]*T[0] + M[0][2]*T[1];
    res[n-1] = M[n-1][0]*T[n-2] + M[n-1][1]*T[n-1] + M[n-1][2]*T[0];
    return res;
}

// Build the tridiagonal matrix with Dirichlet boundary conditions
vector<vector<double>> build_tridiagonal(int n){
    vector<vector<double>> A(n); // Creates one column
    for(uint i=1; i<n-1; i++){
        A[i].resize(n, 0.0); // Creates other elements of the row
        // Fill row
        A[i][i-1] = 1.;
        A[i][i] = -2.;
        A[i][i+1] = 1.;
    }
    // Fix first and last rows separately
    A[0].resize(n, 0.0);
    A[0][0] = 1.0;
    A[n-1].resize(n, 0.0);
    A[n-1][n-1] = 1.0;
    return A;
}

// Build the matrix of 3 vectors starting from the tridiagonal matrix
vector<vector<double>> build_trivector(vector<vector<double>> A){
    int n = A[0].size();
    vector<vector<double>> M(n);
    for(uint i=1; i<n-1; i++){
        M[i].resize(3, 0.0);
        M[i][0] = A[i][i-1];
        M[i][1] = A[i][i];
        M[i][2] = A[i][i+1];
    }
    // Fix first and last rows separately
    M[0].resize(3, 0.0);
    M[0][0] = A[0][n-1];
    M[0][1] = A[0][0];
    M[0][2] = A[0][1];
    M[n-1].resize(3, 0.0);
    M[n-1][0] = A[n-1][n-2];
    M[n-1][1] = A[n-1][n-1];
    M[n-1][2] = A[n-1][0];
    return M;
}

// Gauss-Jordan method (forward elimination backward subsitution)
vector<double> gauss_jordan(vector<vector<double>> X, vector<double> b){
    int n = X[0].size();
    double c; // auxiliary variable useful when swapping rows
    vector<double> sol(n, 0.0); // solution vector

    vector<double> indices;
    for(int i=0; i<N; i++){
        indices.push_back(i);
    }

    // Elimination
    for(uint i=1; i<n; i++){
        for(uint j=0; j<i; j++){
            if (X[i][j] != 0){
                c = (double) -X[i][j]/X[j][j];
                for(uint k=j; k<n; k++){
                    X[i][k] += (double) c*X[j][k];
                    //print_vector(X);
                }
                b[i] += (double) c*b[j];
            }
        }
        // If this row is in the wrong position (i.e. element on the diagonal is 0) put it at the end and redo with the next row
        if (X[i][i] == 0){
            swap_ranges( (X.begin()+i)->begin(), (X.begin()+i)->end(), (X.end()-1)->begin() ); // swap this row and the last one
            c = b[i];
            b[i] = b[n-1];
            b[n-1] = c;
            indices[i] = indices[n-1];
            indices[n-1] = indices[i];
            i -= 1; // re-do this row index with the new row
        }
    }

    // Substitution
    for(int i=n-1; i>=0; i--){
        sol[i] = (double) b[i]/X[i][i];
        for(uint j=i+1; j<n; j++){
            sol[i] -= (double) X[i][j]/X[i][i]*sol[j];
        }
    }

    return sol;
}

// Decomposes matrix into diagonal, upper triangular, lower triangular, in order to perform Jacobi iteration
void DLU_decomposition(vector<vector<double>> &A, vector<double> &D, vector<vector<double>> &L, vector<vector<double>> &U){
    uint n = A.size();
    for(uint i=0; i<n; i++){
        D[i] = A[i][i];
        for(uint j=0; j<n; j++){
            if(i>j) L[i][j] = -A[i][j];   
            else if (i<j) U[i][j] = -A[i][j];
        }
    }
}

void Jacobi_iteration(vector<double> D, vector<vector<double>> L, vector<vector<double>> U, vector<double> &x, vector<double> b){
    int n = x.size();
    double s;
    vector<double> delta(n, 0.0);
    for(int i=0; i<n; i++){
        delta[i] = (double) 1./D[i]*b[i]; 
        s = 0.0;
        for(int j=0; j<n; j++){
            s += (double) U[i][j]*x[j];
            s += (double) L[i][j]*x[j];
        }
        delta[i] += (double) 1./D[i]*s;
    }


    jacobifile << delta[0];
    x[0] = delta[0];
    for(int i=1; i<N; i++) {x[i] = delta[i]; jacobifile << "," << delta[i];}
    jacobifile << endl;

}

// Print 1D vector
void print_vector(vector<double> x){
    for(uint i=0; i<x.size(); i++){
        cout << x[i] << " ";
    }
    cout << endl << endl;
}

// Print 2D vector (overrides 1D)
void print_vector(vector<vector<double>> X){
    for(uint i=0; i<X.size(); i++){
        for(uint j=0; j<X[i].size(); j++){
            cout << showpos << X[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}