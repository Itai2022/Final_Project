
#ifndef PMSC_EXCHANGE_H
#define PMSC_EXCHANGE_H

#include <cmath>
#include <vector>

inline std::vector<bool> compute_send_first(int number_of_processes, int current_process)
{
    // TODO: implement
    std::vector<int> state(number_of_processes);
    std::vector<bool> send(number_of_processes);
    send[0] = 0;

    for(int jump = 1; jump < number_of_processes; jump++)
    {
        std::fill(state.begin(), state.end(), 0);
        for(int process = 0; process < state.size(); process++)
        {
            if(state[process] == 0)
            {
                state[process] = 1;
                int last_position = process;
                int position = (process + jump) % number_of_processes;
                int alternate = -1;
                while(state[position] == 0)
                {
                    state[position] = alternate;
                    last_position = position;
                    position = (position + jump) % number_of_processes;
                    alternate *= -1;
                }
                if(state[position] != alternate)
                {
                    state[last_position] = -1;
                }
            }
        }
        send[jump] = (state[current_process] == 1 ? true : false);
    }
    return send;
}

#endif // PMSC_EXCHANGE_H
