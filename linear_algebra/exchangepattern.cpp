#include "exchangepattern.h"

ExchangePattern::ExchangePattern(std::vector<int> neighboring_processes,
                                 std::vector<std::vector<int>> receive_indices,
                                 std::vector<std::vector<int>> send_indices)
{
    neighboring_processes_ = neighboring_processes;
    receive_indices_ = receive_indices;
    send_indices_ = send_indices;
}

const std::vector<int>& ExchangePattern::neighboring_processes() const
{
    return neighboring_processes_;
}
const std::vector<std::vector<int>>& ExchangePattern::receive_indices() const
{
    return receive_indices_;
}
const std::vector<std::vector<int>>& ExchangePattern::send_indices() const
{
    return send_indices_;
}

std::ostream& operator<<(std::ostream& os, const ExchangePattern& ex)
{
    os << "neighboring_process:{ ";
    for(const auto& neigh : ex.neighboring_processes())
    {
        os << neigh << " ";
    }
    os << "}" << std::endl;
    os << "receive_indices:{";
    for(const auto& recv : ex.receive_indices())
    {
        os << "{ ";
        for(const auto indices : recv)
        {
            os << indices << " ";
        }
        os << "}";
    }
    os << " }" << std::endl;
    os << "send_indicies:{";
    for(const auto& send : ex.send_indices())
    {
        os << "{ ";
        for(const auto indices : send)
        {
            os << indices << " ";
        }
        os << "}";
    }
    os << " }" << std::endl;
    return os;
}