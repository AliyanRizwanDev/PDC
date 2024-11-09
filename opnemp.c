#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Define matrix dimensions
#define N 10

void initialize_matrix(double matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = rand() % 10; // Random values between 0 and 9
        }
    }
}

void print_matrix(double matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%lf ", matrix[i][j]);
        }
        printf("\n");
    }
}

void matrix_multiplication(double matA[N][N], double matB[N][N], double result[N][N]) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = 0;
            for (int k = 0; k < N; k++) {
                result[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
}

void matrix_addition(double matA[N][N], double matB[N][N], double result[N][N]) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = matA[i][j] + matB[i][j];
        }
    }
}

void matrix_subtraction(double matA[N][N], double matB[N][N], double result[N][N]) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = matA[i][j] - matB[i][j];
        }
    }
}

void matrix_transpose(double mat[N][N], double result[N][N]) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[j][i] = mat[i][j];
        }
    }
}

double determinant(double mat[N][N]) {
    double det = 0;
    // Implementing 3x3 determinant for simplicity (you can generalize for NxN)
    det = mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
        - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
        + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
    return det;
}

int is_symmetric(double mat[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (mat[i][j] != mat[j][i]) {
                return 0; // Not symmetric
            }
        }
    }
    return 1; // Symmetric
}

int main() {
    double matA[N][N], matB[N][N], result[N][N];
    
    // Initialize matrices with random values
    initialize_matrix(matA);
    initialize_matrix(matB);

    printf("Matrix A:\n");
    print_matrix(matA);
    printf("\nMatrix B:\n");
    print_matrix(matB);

    double start_time = omp_get_wtime();

    // Perform matrix operations
    matrix_multiplication(matA, matB, result);

    double end_time = omp_get_wtime();

    printf("\nResult Matrix after Multiplication:\n");
    print_matrix(result);

    // Perform addition of matA and matB
    matrix_addition(matA, matB, result);
    printf("\nResult Matrix after Addition:\n");
    print_matrix(result);

    // Perform subtraction of matA and matB
    matrix_subtraction(matA, matB, result);
    printf("\nResult Matrix after Subtraction:\n");
    print_matrix(result);

    // Calculate and print the transpose of matA
    double transpose[N][N];
    matrix_transpose(matA, transpose);
    printf("\nTranspose of Matrix A:\n");
    print_matrix(transpose);

    // Calculate and print the determinant of matA (3x3 example)
    double det = determinant(matA);
    printf("\nDeterminant of Matrix A: %lf\n", det);

    // Check if matA is symmetric
    int symmetric = is_symmetric(matA);
    if (symmetric) {
        printf("\nMatrix A is symmetric.\n");
    } else {
        printf("\nMatrix A is not symmetric.\n");
    }

    printf("\nExecution Time: %f seconds\n", end_time - start_time);

    return 0;
}
