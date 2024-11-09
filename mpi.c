#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int userId;
    char title[100];
    int duration;
    char genre[50];
} Record;

void processRecords(Record *records, int numRecords, int *results) {
    // Process records to count durations in different ranges
    for (int i = 0; i < numRecords; i++) {
        if (records[i].duration >= 1 && records[i].duration < 30) {
            results[0]++;
        } else if (records[i].duration >= 30 && records[i].duration < 120) {
            results[1]++;
        } else if (records[i].duration >= 120 && records[i].duration < 240) {
            results[2]++;
        } else if (records[i].duration >= 240) {
            results[3]++;
        }
    }
}

void initMPI(int argc, char *argv[], int *numProcs, int *rank) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
}

void finalizeMPI() {
    MPI_Finalize();
}

void loadData(int rank, Record **data, int *totalRecords) {
    if (rank == 0) {
        // Load dataset from a file
        FILE *file = fopen("dataset.txt", "r");
        if (!file) {
            printf("Error: Unable to open the dataset file.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fscanf(file, "%d", totalRecords);
        *data = (Record *)malloc(*totalRecords * sizeof(Record));

        for (int i = 0; i < *totalRecords; i++) {
            fscanf(file, "%d %s %d %s", &(*data)[i].userId, (*data)[i].title, &(*data)[i].duration, (*data)[i].genre);
        }

        fclose(file);
    }
}

void distributeData(Record *data, int totalRecords, Record **localData, int localRecords) {
    MPI_Scatter(data, localRecords * sizeof(Record), MPI_BYTE,
                *localData, localRecords * sizeof(Record), MPI_BYTE,
                0, MPI_COMM_WORLD);
}

void gatherResults(int *localCounts, int *globalCounts, int countSize) {
    MPI_Reduce(localCounts, globalCounts, countSize, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
    int numProcs, rank;
    initMPI(argc, argv, &numProcs, &rank);

    if (numProcs != 4) {
        if (rank == 0) {
            printf("This program requires exactly 4 MPI tasks.\n");
        }
        finalizeMPI();
        return 0;
    }

    int totalRecords;
    Record *data = NULL;
    loadData(rank, &data, &totalRecords);

    MPI_Bcast(&totalRecords, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int localRecords = totalRecords / numProcs;
    Record *localData = (Record *)malloc(localRecords * sizeof(Record));

    distributeData(data, totalRecords, &localData, localRecords);

    int localCounts[4] = {0, 0, 0, 0};
    processRecords(localData, localRecords, localCounts);

    int globalCounts[4] = {0, 0, 0, 0};
    gatherResults(localCounts, globalCounts, 4);

    if (rank == 0) {
            printf("Count of users who watched content with durations:\n");
            printf("1 to 30 minutes: %d\n", globalCounts[0]);
            printf("30 to 120 minutes: %d\n", globalCounts[1]);
            printf("120 to 240 minutes: %d\n", globalCounts[2]);
            printf("Exceeding 240 minutes: %d\n", globalCounts[3]); }
            
    if (rank == 0) {
        free(data);
    }
    free(localData);
    finalizeMPI();
    return 0;
}
