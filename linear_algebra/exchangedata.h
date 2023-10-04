#pragma once

#include "exchange.h"
#include "exchangepattern.h"
#include "vector.h"
#include <cassert>
#include <type_traits>



template<typename T>
class ExchangeData
{
private:
    ExchangePattern exchange_pattern_;
    std::vector<std::vector<T>> data_per_neighboring_process_;

public:
    ExchangeData() = default;
    ~ExchangeData() = default;
    explicit ExchangeData(const ExchangePattern& exchange_pattern, std::vector<std::vector<T>> data_per_neighboring_process);

    const T& get(int owner_rank, int global_index) const;
};

template<typename T>
ExchangeData<T>::ExchangeData(const ExchangePattern& exchange_pattern, std::vector<std::vector<T>> data_per_neighboring_process)
{
    exchange_pattern_ = exchange_pattern;
    data_per_neighboring_process_ = data_per_neighboring_process;
}

template<typename T>
const T& ExchangeData<T>::get(int owner_rank, int global_index) const
{

    const auto& result_rank = std::find(exchange_pattern_.neighboring_processes().begin(),
                                        exchange_pattern_.neighboring_processes().end(), owner_rank);
    const auto rank_index = (result_rank - exchange_pattern_.neighboring_processes().begin());

    const auto& result_index = std::find(exchange_pattern_.receive_indices()[rank_index].begin(),
                                         exchange_pattern_.receive_indices()[rank_index].end(), global_index);
    const auto local_index = (result_index - exchange_pattern_.receive_indices()[rank_index].begin());

    return data_per_neighboring_process_[rank_index][local_index];
}

template<typename T>
ExchangeData<T> exchange_vector_data(const ExchangePattern& exchange_pattern, const Vector<T>& vector)
{
    MPI_Comm communicator = vector.partition().communicator();
    int number_of_processes;
    int myrank;
    MPI_Comm_size(communicator, &number_of_processes);
    MPI_Comm_rank(communicator, &myrank);

    const auto& neighboring_processes = exchange_pattern.neighboring_processes();
    const auto& send_indices = exchange_pattern.send_indices();
    const auto& receive_indices = exchange_pattern.receive_indices();
    const auto number_of_neighbor = exchange_pattern.neighboring_processes().size();
    const auto& partition = vector.partition();
    std::vector<std::vector<T>> data_per_neighboring_process(number_of_neighbor);

    const auto send_first = compute_send_first(number_of_processes, myrank);

    for(int jump = 1; jump < number_of_processes; jump++)
    {
        int receiver = (myrank + jump) % number_of_processes;
        int sender = (myrank + number_of_processes - jump) % number_of_processes;

        auto result_recv = std::find(neighboring_processes.begin(), neighboring_processes.end(), receiver);
        auto result_send = std::find(neighboring_processes.begin(), neighboring_processes.end(), sender);
        if(send_first[jump])
        {
            if(result_recv != neighboring_processes.end())
            {
                // send
                std::vector<T> send_data;
                const auto receiver_index = result_recv - neighboring_processes.begin();
                const auto send_size = send_indices[receiver_index].size();
                for(int i = 0; i < send_size; i++)
                {
                    send_data.push_back(vector[partition.to_local_index(send_indices[receiver_index][i])]);
                }
                MPI_Send(&send_data[0], send_size, mpi_get_type<T>(), receiver, 99, communicator);
            }
            if(result_send != neighboring_processes.end())
            {
                // receive
                const auto sender_index = result_send - neighboring_processes.begin();
                const auto receive_size = receive_indices[sender_index].size();
                std::vector<T> receive_data(receive_size);
                MPI_Recv(&receive_data[0], receive_size, mpi_get_type<T>(), sender, 99, communicator, MPI_STATUS_IGNORE);
                data_per_neighboring_process[sender_index] = receive_data;
            }
        }
        else
        {
            if(result_send != neighboring_processes.end())
            {
                // receive
                const auto sender_index = result_send - neighboring_processes.begin();
                const auto receive_size = receive_indices[sender_index].size();
                std::vector<T> receive_data(receive_size);
                MPI_Recv(&receive_data[0], receive_size, mpi_get_type<T>(), sender, 99, communicator, MPI_STATUS_IGNORE);
                data_per_neighboring_process[sender_index] = receive_data;
            }
            if(result_recv != neighboring_processes.end())
            {
                // send
                std::vector<T> send_data;
                const auto receiver_index = result_recv - neighboring_processes.begin();
                const auto send_size = send_indices[receiver_index].size();
                for(int i = 0; i < send_size; i++)
                {
                    send_data.push_back(vector[partition.to_local_index(send_indices[receiver_index][i])]);
                }
                MPI_Send(&send_data[0], send_size, mpi_get_type<T>(), receiver, 99, communicator);
            }
        }
        /*
        if(result_recv != exchange_pattern.neighboring_processes().end() || result_send != exchange_pattern.neighboring_processes().end())
        {
            // send
            std::vector<T> send_data;
            auto receiver_index = result_recv - exchange_pattern.neighboring_processes().begin();
            int send_size = (int)exchange_pattern.send_indices()[receiver_index].size();
            for(int i = 0; i < send_size; i++)
            {
                send_data.push_back(vector[vector.partition().to_local_index(exchange_pattern.send_indices()[receiver_index][i])]);
            }

            // receive
            auto sender_index = result_send - exchange_pattern.neighboring_processes().begin();
            int receive_size = (int)exchange_pattern.receive_indices()[sender_index].size();
            std::vector<T> receive_data(receive_size);
            std::cout << " should print" << std::endl;
            if(compute_send_first(number_of_processes, myrank)[jump] == true)
            {
                MPI_Send(&send_data[0], send_size, mpi_get_type<T>(), receiver, 99, communicator);
                MPI_Recv(&receive_data[0], receive_size, mpi_get_type<T>(), sender, 99, communicator, MPI_STATUS_IGNORE);
                data_per_neighboring_process[sender_index] = receive_data;
            }
            else
            {
                MPI_Recv(&receive_data[0], receive_size, mpi_get_type<T>(), sender, 99, communicator, MPI_STATUS_IGNORE);
                data_per_neighboring_process[sender_index] = receive_data;
                MPI_Send(&send_data[0], send_size, mpi_get_type<T>(), receiver, 99, communicator);
            }
        }
        */
    }
    return ExchangeData<T>(exchange_pattern, data_per_neighboring_process);
}
