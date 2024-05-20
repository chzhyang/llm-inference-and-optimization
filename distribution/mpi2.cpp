// #include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
// void mpi_finalize() {
//     int is_finalized = 0;
//     MPI_Finalized(&is_finalized);

//     if (!is_finalized) {
//         MPI_Finalize();
//     }
// }

int main(int argc, char* argv[]) {

    int size = 0;
    int rank = 0;
    

    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len = 0;
    MPI_Get_processor_name(processor_name, &name_len);

    // atexit(mpi_finalize);
    printf("processor_name: %s rank:%d size: %d\n", processor_name,rank,size);
    // std::cout << "rank: "<< rank <<"  " << "size: " << size << "\n";
    // std::cout << "MPI_COMM_WORLD: " << MPI_COMM_WORLD <<"\n";

    sleep(15);
    printf("exit!");
    MPI_Finalize();
}